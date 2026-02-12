/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![cfg(test)]

use lsp_types::HoverContents;

use crate::lsp::wasm::hover::get_hover;
use crate::test::util::TestEnv;
use crate::test::util::extract_cursors_for_test;
use crate::testcase;

/// Creates a TestEnv configured as a blender extension with bpy stubs.
///
/// The init module is `my_addon` (matching the blender_init_module config).
/// Callers should add the init module content and any other files as needed.
fn blender_env() -> TestEnv {
    let mut env = TestEnv::new().with_blender_init_module("my_addon");

    // bpy/__init__.pyi
    env.add_with_path(
        "bpy",
        "bpy/__init__.pyi",
        "from . import types as types\nfrom . import props as props\n",
    );

    // bpy/types/__init__.pyi
    env.add_with_path(
        "bpy.types",
        "bpy/types/__init__.pyi",
        "class Scene: ...\nclass Object: ...\nclass PropertyGroup: ...\n",
    );

    // bpy/props/__init__.pyi
    env.add_with_path(
        "bpy.props",
        "bpy/props/__init__.pyi",
        r#"
def StringProperty(**kwargs: object) -> str: ...
def IntProperty(**kwargs: object) -> int: ...
def FloatProperty(**kwargs: object) -> float: ...
def BoolProperty(**kwargs: object) -> bool: ...
"#,
    );

    env
}

testcase!(
    test_basic_string_property,
    {
        let mut env = blender_env();
        env.add(
            "my_addon",
            r#"
import bpy

def register():
    bpy.types.Scene.my_prop = bpy.props.StringProperty()
"#,
        );
        env
    },
    r#"
import bpy
from typing_extensions import assert_type

def check(scene: bpy.types.Scene) -> None:
    assert_type(scene.my_prop, str)
"#,
);

testcase!(
    test_int_property,
    {
        let mut env = blender_env();
        env.add(
            "my_addon",
            r#"
import bpy

def register():
    bpy.types.Scene.count = bpy.props.IntProperty()
"#,
        );
        env
    },
    r#"
import bpy
from typing_extensions import assert_type

def check(scene: bpy.types.Scene) -> None:
    assert_type(scene.count, int)
"#,
);

testcase!(
    test_float_property,
    {
        let mut env = blender_env();
        env.add(
            "my_addon",
            r#"
import bpy

def register():
    bpy.types.Scene.value = bpy.props.FloatProperty()
"#,
        );
        env
    },
    r#"
import bpy
from typing_extensions import assert_type

def check(scene: bpy.types.Scene) -> None:
    assert_type(scene.value, float)
"#,
);

testcase!(
    test_bool_property,
    {
        let mut env = blender_env();
        env.add(
            "my_addon",
            r#"
import bpy

def register():
    bpy.types.Scene.flag = bpy.props.BoolProperty()
"#,
        );
        env
    },
    r#"
import bpy
from typing_extensions import assert_type

def check(scene: bpy.types.Scene) -> None:
    assert_type(scene.flag, bool)
"#,
);

testcase!(
    test_imported_class_from_import,
    {
        let mut env = blender_env();
        env.add(
            "my_addon",
            r#"
from bpy.types import Scene
import bpy

def register():
    Scene.my_prop = bpy.props.StringProperty()
"#,
        );
        env
    },
    r#"
import bpy
from typing_extensions import assert_type

def check(scene: bpy.types.Scene) -> None:
    assert_type(scene.my_prop, str)
"#,
);

testcase!(
    test_imported_module_dotted,
    {
        let mut env = blender_env();
        env.add(
            "my_addon",
            r#"
from bpy import types
import bpy

def register():
    types.Scene.my_prop = bpy.props.StringProperty()
"#,
        );
        env
    },
    r#"
import bpy
from typing_extensions import assert_type

def check(scene: bpy.types.Scene) -> None:
    assert_type(scene.my_prop, str)
"#,
);

testcase!(
    test_unregistered_property_errors,
    {
        let mut env = blender_env();
        env.add(
            "my_addon",
            r#"
import bpy

def register():
    bpy.types.Scene.my_prop = bpy.props.StringProperty()
"#,
        );
        env
    },
    r#"
import bpy

def check(scene: bpy.types.Scene) -> None:
    x = scene.nonexistent  # E: has no attribute `nonexistent`
"#,
);

/// Hover test: verifies that hovering over a dynamically registered blender
/// property shows the correct type in IDE hover results.
#[test]
fn test_hover_shows_blender_property_type() {
    let mut env = blender_env();
    env.add(
        "my_addon",
        r#"
import bpy

def register():
    bpy.types.Scene.my_prop = bpy.props.StringProperty()
"#,
    );
    let test_code = r#"
import bpy

def check(scene: bpy.types.Scene) -> None:
    scene.my_prop
#         ^
"#;
    env.add_with_path("main", "main.py", test_code);
    let (state, handle) = env.to_state();
    let cursors = extract_cursors_for_test(test_code);
    assert!(!cursors.is_empty(), "Expected at least one cursor position");
    let transaction = state.transaction();
    let h = handle("main");
    let hover = get_hover(&transaction, &h, cursors[0], false);
    let hover_text = match hover {
        Some(lsp_types::Hover {
            contents: HoverContents::Markup(markup),
            ..
        }) => markup.value,
        _ => String::new(),
    };
    assert!(
        hover_text.contains("str"),
        "Expected hover to show `str` type for blender property, got: {hover_text}"
    );
}

/// Hover test: verifies that get_type_at resolves the type for a dynamically
/// registered blender property.
#[test]
fn test_get_type_at_blender_property() {
    let mut env = blender_env();
    env.add(
        "my_addon",
        r#"
import bpy

def register():
    bpy.types.Scene.my_prop = bpy.props.StringProperty()
    bpy.types.Scene.count = bpy.props.IntProperty()
"#,
    );
    let test_code = r#"
import bpy

def check(scene: bpy.types.Scene) -> None:
    scene.my_prop
#         ^
    scene.count
#         ^
"#;
    env.add_with_path("main", "main.py", test_code);
    let (state, handle) = env.to_state();
    let cursors = extract_cursors_for_test(test_code);
    assert_eq!(cursors.len(), 2, "Expected exactly 2 cursor positions");
    let transaction = state.transaction();
    let h = handle("main");

    // Check my_prop resolves to str
    let ty1 = transaction.get_type_at(&h, cursors[0]);
    assert!(
        ty1.is_some(),
        "Expected get_type_at to return a type for blender property `my_prop`"
    );
    let ty1_str = format!("{}", ty1.unwrap());
    assert!(
        ty1_str.contains("str"),
        "Expected `my_prop` to resolve to `str`, got: {ty1_str}"
    );

    // Check count resolves to int
    let ty2 = transaction.get_type_at(&h, cursors[1]);
    assert!(
        ty2.is_some(),
        "Expected get_type_at to return a type for blender property `count`"
    );
    let ty2_str = format!("{}", ty2.unwrap());
    assert!(
        ty2_str.contains("int"),
        "Expected `count` to resolve to `int`, got: {ty2_str}"
    );
}
