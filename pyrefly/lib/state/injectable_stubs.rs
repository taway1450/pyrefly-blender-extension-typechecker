/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use pyrefly_python::module_name::ModuleName;
use pyrefly_python::module_path::ModulePath;
use pyrefly_python::module_path::ModulePathDetails;
use pyrefly_python::sys_info::PythonVersion;
use ruff_python_ast::ModModule;
use ruff_python_ast::PySourceType;
use ruff_python_ast::Stmt;
use ruff_python_ast::name::Name;

use crate::error::collector::ErrorCollector;
use crate::module::parse::module_parse;

fn injectable_stub_path(root: &Path, module: ModuleName, is_init: bool) -> Option<PathBuf> {
    let components = module.components();
    if components.is_empty() {
        return None;
    }

    let mut path = root.join(".injectable_stubs");
    for component in components {
        path.push(component.as_str());
    }
    if is_init {
        path.push("__init__.pyi");
    } else {
        path.set_extension("pyi");
    }
    Some(path)
}

fn top_level_function_name(stmt: &Stmt) -> Option<Name> {
    match stmt {
        Stmt::FunctionDef(def) => Some(def.name.id.clone()),
        _ => None,
    }
}

fn class_member_name(stmt: &Stmt) -> Option<Name> {
    match stmt {
        Stmt::FunctionDef(def) => Some(def.name.id.clone()),
        Stmt::ClassDef(def) => Some(def.name.id.clone()),
        Stmt::AnnAssign(assign) => assign.target.as_name_expr().map(|name| name.id.clone()),
        Stmt::Assign(assign) => {
            if assign.targets.len() == 1 {
                assign.targets[0].as_name_expr().map(|name| name.id.clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn merge_class_body(base_body: &mut Vec<Stmt>, injectable_body: &[Stmt]) {
    let mut member_indexes = std::collections::HashMap::new();
    for (index, stmt) in base_body.iter().enumerate() {
        if let Some(name) = class_member_name(stmt) {
            member_indexes.insert(name, index);
        }
    }

    for injectable_stmt in injectable_body {
        if let Some(name) = class_member_name(injectable_stmt) {
            if let Some(index) = member_indexes.get(&name).copied() {
                base_body[index] = injectable_stmt.clone();
            } else {
                let index = base_body.len();
                base_body.push(injectable_stmt.clone());
                member_indexes.insert(name, index);
            }
        } else {
            base_body.push(injectable_stmt.clone());
        }
    }
}

fn merge_injectable_stub_into_ast(base: &mut ModModule, injectable: ModModule) {
    for injectable_stmt in injectable.body {
        match &injectable_stmt {
            Stmt::ClassDef(injectable_class) => {
                if let Some(Stmt::ClassDef(base_class)) = base.body.iter_mut().find(
                    |stmt| {
                        matches!(stmt, Stmt::ClassDef(class) if class.name.id == injectable_class.name.id)
                    },
                ) {
                    merge_class_body(&mut base_class.body, &injectable_class.body);
                } else {
                    base.body.push(injectable_stmt.clone());
                }
            }
            _ => {
                if let Some(name) = top_level_function_name(&injectable_stmt) {
                    if let Some(index) = base
                        .body
                        .iter()
                        .position(|stmt| top_level_function_name(stmt).is_some_and(|x| x == name))
                    {
                        base.body[index] = injectable_stmt.clone();
                    } else {
                        base.body.push(injectable_stmt.clone());
                    }
                }
            }
        }
    }
}

fn module_is_loaded_from_injectable_stub(
    loaded_module_path: &ModulePath,
    injectable_stub_path: &Path,
) -> bool {
    match loaded_module_path.details() {
        ModulePathDetails::FileSystem(path) => path.as_path() == injectable_stub_path,
        _ => false,
    }
}

pub fn merge_injectable_stub_if_present(
    ast: &mut ModModule,
    loaded_module_path: &ModulePath,
    root: Option<&Path>,
    module: ModuleName,
    is_init: bool,
    version: PythonVersion,
    errors: &ErrorCollector,
) {
    if let Some(root) = root
        && let Some(path) = injectable_stub_path(root, module, is_init)
        && !module_is_loaded_from_injectable_stub(loaded_module_path, &path)
        && let Ok(contents) = fs::read_to_string(path)
    {
        let injectable = module_parse(&contents, version, PySourceType::Stub, errors);
        merge_injectable_stub_into_ast(ast, injectable);
    }
}
