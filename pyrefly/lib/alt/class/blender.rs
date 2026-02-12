/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Blender dynamic property resolution.
//!
//! When a Blender extension's `register()` function assigns properties like
//! `bpy.types.Scene.my_prop = bpy.props.StringProperty()`, this module
//! resolves attribute lookups on class instances by checking synthetic
//! exports in the blender init module.

use pyrefly_types::class::Class;
use pyrefly_types::types::Type;
use ruff_python_ast::name::Name;

use crate::alt::answers::LookupAnswer;
use crate::alt::answers_solver::AnswersSolver;
use crate::binding::binding::KeyExport;
use crate::export::blender::blender_prop_export_name;

impl<'a, Ans: LookupAnswer> AnswersSolver<'a, Ans> {
    /// Looks up a dynamically registered blender property on a class.
    ///
    /// Returns `Some(ty)` if the blender init module has a synthetic export
    /// matching the class and attribute name; `None` otherwise.
    pub fn lookup_blender_property(&self, class: &Class, attr_name: &Name) -> Option<Type> {
        let blender_init = self.bindings().blender_init_module()?;
        let target_module = class.module_name();
        let target_class = class.name();
        let export_name = blender_prop_export_name(target_module, target_class, attr_name);
        if !self.exports.export_exists(blender_init, &export_name) {
            return None;
        }
        let ty = self.get_from_export(blender_init, None, &KeyExport(export_name));
        Some(ty.arc_clone())
    }
}
