/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Blender extension support: scans the `register()` function in a Blender
//! extension's `__init__.py` for dynamic property assignments like
//! `bpy.types.Scene.my_prop = bpy.props.StringProperty()` and creates
//! synthetic definitions so they can be resolved as attributes.

use pyrefly_python::module_name::ModuleName;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprName;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;
use ruff_text_size::TextRange;

use crate::export::definitions::DefinitionStyle;
use crate::export::definitions::Definitions;

/// A single dynamic Blender property registration found in `register()`.
#[derive(Debug, Clone)]
pub struct BlenderPropertyRegistration {
    /// The module containing the target class (e.g., `bpy.types`).
    pub target_module: ModuleName,
    /// The name of the target class (e.g., `Scene`).
    pub target_class: Name,
    /// The property name being assigned (e.g., `my_prop`).
    pub prop_name: Name,
    /// The RHS call expression (e.g., `bpy.props.StringProperty()`), cloned from AST.
    pub rhs_expr: Expr,
    /// The text range of the assignment target.
    pub range: TextRange,
}

/// Generates a deterministic synthetic export name for a blender property registration.
/// Format: `__blender_prop__{target_module}__{target_class}__{prop_name}`
pub fn blender_prop_export_name(
    target_module: ModuleName,
    target_class: &Name,
    prop_name: &Name,
) -> Name {
    Name::new(format!(
        "__blender_prop__{}__{}__{}",
        target_module.as_str(),
        target_class.as_str(),
        prop_name.as_str()
    ))
}

/// Scans the top-level statements of a blender init module for a `register()`
/// function definition and extracts dynamic property assignments from its body.
pub fn scan_register_for_blender_properties(
    stmts: &[Stmt],
    definitions: &Definitions,
    module_name: ModuleName,
) -> Vec<BlenderPropertyRegistration> {
    let mut registrations = Vec::new();

    for stmt in stmts {
        if let Stmt::FunctionDef(func_def) = stmt
            && func_def.name.as_str() == "register"
        {
            scan_body_for_property_assignments(
                &func_def.body,
                definitions,
                module_name,
                &mut registrations,
            );
            break;
        }
    }

    registrations
}

/// Walks the body of register() looking for assignments of the form
/// `<target_class_expr>.<prop_name> = <rhs_call_expr>`.
fn scan_body_for_property_assignments(
    body: &[Stmt],
    definitions: &Definitions,
    module_name: ModuleName,
    registrations: &mut Vec<BlenderPropertyRegistration>,
) {
    for stmt in body {
        let Stmt::Assign(assign) = stmt else {
            continue;
        };
        // Only single-target assignments
        if assign.targets.len() != 1 {
            continue;
        }
        let target = &assign.targets[0];
        // Target must be an attribute access: `X.prop_name`
        let Expr::Attribute(outer_attr) = target else {
            continue;
        };
        let prop_name = &outer_attr.attr;
        let class_expr = &*outer_attr.value;

        // RHS must be a call expression (e.g., `bpy.props.StringProperty(...)`)
        if !assign.value.is_call_expr() {
            continue;
        }

        // Resolve the class expression to a (module, class_name) pair
        if let Some((target_module, target_class)) =
            resolve_target_class(class_expr, definitions, module_name)
        {
            registrations.push(BlenderPropertyRegistration {
                target_module,
                target_class,
                prop_name: prop_name.id.clone(),
                rhs_expr: (*assign.value).clone(),
                range: outer_attr.range,
            });
        }
    }
}

/// Resolves an expression to a `(module_name, class_name)` pair by
/// examining the AST structure and looking up imports in the definitions.
///
/// Supported patterns:
/// - `bpy.types.Scene` → `(bpy.types, Scene)` (dotted attribute chain)
/// - `Scene` (from `from bpy.types import Scene`) → `(bpy.types, Scene)`
/// - `types.Scene` (from `from bpy import types`) → `(bpy.types, Scene)`
/// - `MyClass` (locally defined) → `(current_module, MyClass)`
fn resolve_target_class(
    expr: &Expr,
    definitions: &Definitions,
    module_name: ModuleName,
) -> Option<(ModuleName, Name)> {
    match expr {
        // Case: `SomeClass` (bare name)
        Expr::Name(ExprName { id, .. }) => resolve_bare_name(id, definitions, module_name),

        // Case: `something.ClassName` (attribute access)
        Expr::Attribute(ExprAttribute { value, attr, .. }) => {
            let class_name = attr.id.clone();
            // Resolve the prefix to a module name
            let prefix_module = resolve_expr_to_module(value, definitions)?;
            Some((prefix_module, class_name))
        }

        _ => None,
    }
}

/// Resolves a bare name to a `(module, class)` pair using import definitions.
fn resolve_bare_name(
    name: &Name,
    definitions: &Definitions,
    module_name: ModuleName,
) -> Option<(ModuleName, Name)> {
    let def = definitions.definitions.get(name)?;
    match &def.style {
        // `from bpy.types import Scene`
        DefinitionStyle::Import(source_module) => Some((*source_module, name.clone())),
        // `from bpy.types import Scene as Scene`
        DefinitionStyle::ImportAsEq(source_module) => Some((*source_module, name.clone())),
        // `from bpy.types import OrigName as Scene`
        DefinitionStyle::ImportAs(source_module, original_name) => {
            Some((*source_module, original_name.clone()))
        }
        // Locally defined class
        DefinitionStyle::Unannotated(_) | DefinitionStyle::Annotated(..) => {
            Some((module_name, name.clone()))
        }
        _ => None,
    }
}

/// Resolves an expression to a module name by following the import chain.
/// Handles dotted chains like `bpy.types` or single names like `types`.
fn resolve_expr_to_module(expr: &Expr, definitions: &Definitions) -> Option<ModuleName> {
    match expr {
        // Base case: `bpy` or `types`
        Expr::Name(ExprName { id, .. }) => {
            let def = definitions.definitions.get(id)?;
            match &def.style {
                // `import bpy` → `bpy` is a module
                DefinitionStyle::ImportModule(m) => Some(*m),
                // `from bpy import types` → `types` refers to submodule `bpy.types`
                DefinitionStyle::Import(source_module) => Some(source_module.append(id)),
                _ => None,
            }
        }

        // Recursive case: `bpy.types` (attribute on a module)
        Expr::Attribute(ExprAttribute { value, attr, .. }) => {
            let parent_module = resolve_expr_to_module(value, definitions)?;
            Some(parent_module.append(&attr.id))
        }

        _ => None,
    }
}
