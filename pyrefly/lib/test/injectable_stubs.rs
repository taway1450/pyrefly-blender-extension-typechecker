/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fs;

use tempfile::tempdir;

use crate::test::util::TestEnv;
use crate::test::util::testcase_for_macro;

#[test]
fn test_injectable_stub_merges_source_module() -> anyhow::Result<()> {
    let mut env = TestEnv::new();
    env.add_with_path(
        "foo",
        "foo.py",
        r#"
class A:
    x: int = 1

    def f(self) -> int:
        return 1

def top() -> int:
    return 1
"#,
    );

    let temp = tempdir()?;
    let root = temp.path();
    fs::create_dir_all(root.join(".injectable_stubs"))?;
    fs::write(
        root.join(".injectable_stubs/foo.pyi"),
        r#"
class A:
    x: str
    y: str

    def f(self) -> str: ...
    def g(self) -> bool: ...

def top() -> str: ...
def added() -> bytes: ...
z: int
"#,
    )?;

    testcase_for_macro(
        env.with_config_source_root(root.to_path_buf()),
        r#"
from typing import assert_type
import foo

assert_type(foo.A.x, str)
assert_type(foo.A.y, str)
assert_type(foo.A().f(), str)
assert_type(foo.A().g(), bool)
assert_type(foo.top(), str)
assert_type(foo.added(), bytes)
foo.z  # E: No attribute `z` in module `foo`
"#,
        file!(),
        line!(),
    )
}

#[test]
fn test_injectable_stub_merges_normal_stub_module() -> anyhow::Result<()> {
    let mut env = TestEnv::new();
    env.add_with_path(
        "foo",
        "foo.py",
        r#"
class A:
    pass

def top() -> int:
    return 1
"#,
    );
    env.add_with_path(
        "foo",
        "foo.pyi",
        r#"
class A:
    x: int

def top() -> int: ...
"#,
    );

    let temp = tempdir()?;
    let root = temp.path();
    fs::create_dir_all(root.join(".injectable_stubs"))?;
    fs::write(
        root.join(".injectable_stubs/foo.pyi"),
        r#"
class A:
    x: str
    y: str

def top() -> str: ...
def added() -> bytes: ...
"#,
    )?;

    testcase_for_macro(
        env.with_config_source_root(root.to_path_buf()),
        r#"
from typing import assert_type
import foo

assert_type(foo.A.x, str)
assert_type(foo.A.y, str)
assert_type(foo.top(), str)
assert_type(foo.added(), bytes)
"#,
        file!(),
        line!(),
    )
}

#[test]
fn test_no_injectable_stub_keeps_existing_behavior() -> anyhow::Result<()> {
    let temp = tempdir()?;
    let root = temp.path();

    let mut env = TestEnv::new();
    env.add_with_path(
        "foo",
        "foo.py",
        r#"
def top() -> int:
    return 1
"#,
    );

    testcase_for_macro(
        env.with_config_source_root(root.to_path_buf()),
        r#"
from typing import assert_type
import foo

assert_type(foo.top(), int)
foo.added()  # E: No attribute `added` in module `foo`
"#,
        file!(),
        line!(),
    )
}

#[test]
fn test_injectable_stub_merges_multifile_project_dependency() -> anyhow::Result<()> {
    let mut env = TestEnv::new();
    env.add_with_path(
        "core.models",
        "core/models.py",
        r#"
class User:
    def id(self) -> int:
        return 1
"#,
    );
    env.add_with_path(
        "core.service",
        "core/service.py",
        r#"
from core.models import User

def fetch() -> str:
    return User().id()
"#,
    );

    let temp = tempdir()?;
    let root = temp.path();
    fs::create_dir_all(root.join(".injectable_stubs/core"))?;
    fs::write(
        root.join(".injectable_stubs/core/models.pyi"),
        r#"
class User:
    def id(self) -> str: ...
    def name(self) -> str: ...
"#,
    )?;

    testcase_for_macro(
        env.with_config_source_root(root.to_path_buf()),
        r#"
from typing import assert_type
from core.models import User
from core.service import fetch

assert_type(User().id(), str)
assert_type(User().name(), str)
assert_type(fetch(), str)
"#,
        file!(),
        line!(),
    )
}

#[test]
fn test_injectable_stub_merges_imported_library_in_multifile_project() -> anyhow::Result<()> {
    let mut env = TestEnv::new();
    env.add_with_path(
        "thirdparty",
        "thirdparty/__init__.py",
        r#"
class Client:
    def version(self) -> int:
        return 1
"#,
    );
    env.add_with_path(
        "app.client",
        "app/client.py",
        r#"
from thirdparty import Client

def read_version() -> str:
    return Client().version()
"#,
    );

    let temp = tempdir()?;
    let root = temp.path();
    fs::create_dir_all(root.join(".injectable_stubs/thirdparty"))?;
    fs::write(
        root.join(".injectable_stubs/thirdparty/__init__.pyi"),
        r#"
class Client:
    def version(self) -> str: ...

def ping() -> bytes: ...
"#,
    )?;

    testcase_for_macro(
        env.with_config_source_root(root.to_path_buf()),
        r#"
from typing import assert_type
from app.client import read_version
from thirdparty import Client, ping

assert_type(Client().version(), str)
assert_type(read_version(), str)
assert_type(ping(), bytes)
"#,
        file!(),
        line!(),
    )
}

#[test]
fn test_injectable_stub_merges_bpy_types_from_site_packages() -> anyhow::Result<()> {
    let temp = tempdir()?;
    let root = temp.path();
    let site_packages = root.join("site-packages");

    fs::create_dir_all(site_packages.join("bpy/types"))?;
    fs::write(
        site_packages.join("bpy/__init__.pyi"),
        r#"
from . import types
"#,
    )?;
    fs::write(
        site_packages.join("bpy/types/__init__.pyi"),
        r#"
class Scene:
    pass
"#,
    )?;

    fs::create_dir_all(root.join(".injectable_stubs/bpy/types"))?;
    fs::write(
        root.join(".injectable_stubs/bpy/types/__init__.pyi"),
        r#"
class Scene:
    added_property: bool
"#,
    )?;

    testcase_for_macro(
        TestEnv::new_with_site_package_path(
            site_packages
                .to_str()
                .expect("site-packages path must be valid UTF-8"),
        )
        .with_config_source_root(root.to_path_buf()),
        r#"
from typing import assert_type
import bpy
import bpy.types

assert_type(bpy.types.Scene.added_property, bool)
"#,
        file!(),
        line!(),
    )
}

#[test]
fn test_injectable_stub_resolves_injectable_only_bpy_ops_submodule() -> anyhow::Result<()> {
    let temp = tempdir()?;
    let root = temp.path();
    let site_packages = root.join("site-packages");

    fs::create_dir_all(site_packages.join("bpy/ops"))?;
    fs::write(
        site_packages.join("bpy/__init__.pyi"),
        r#"
from . import ops
"#,
    )?;
    fs::write(site_packages.join("bpy/ops/__init__.pyi"), "")?;

    fs::create_dir_all(root.join(".injectable_stubs/bpy/ops/my_blender_addon"))?;
    fs::write(
        root.join(".injectable_stubs/bpy/ops/my_blender_addon/__init__.pyi"),
        r#"
def add_custom_workspace(
    execution_context: int | str | None = None,
    undo: bool | None = None,
    /,
) -> set[str]: ...
"#,
    )?;

    testcase_for_macro(
        TestEnv::new_with_site_package_path(
            site_packages
                .to_str()
                .expect("site-packages path must be valid UTF-8"),
        )
        .with_config_source_root(root.to_path_buf()),
        r#"
from typing import assert_type
import bpy.ops.my_blender_addon

assert_type(bpy.ops.my_blender_addon.add_custom_workspace(), set[str])
"#,
        file!(),
        line!(),
    )
}

#[test]
fn test_injectable_only_module_does_not_self_merge() -> anyhow::Result<()> {
    let temp = tempdir()?;
    let root = temp.path();
    let site_packages = root.join("site-packages");

    fs::create_dir_all(site_packages.join("bpy/ops"))?;
    fs::write(
        site_packages.join("bpy/__init__.pyi"),
        r#"
from . import ops
"#,
    )?;
    fs::write(site_packages.join("bpy/ops/__init__.pyi"), "")?;

    fs::create_dir_all(root.join(".injectable_stubs/bpy/ops/my_blender_addon"))?;
    fs::write(
        root.join(".injectable_stubs/bpy/ops/my_blender_addon/__init__.pyi"),
        r#"
class OperatorProps:
    import typing as t

def build() -> OperatorProps: ...
"#,
    )?;

    testcase_for_macro(
        TestEnv::new_with_site_package_path(
            site_packages
                .to_str()
                .expect("site-packages path must be valid UTF-8"),
        )
        .with_config_source_root(root.to_path_buf()),
        r#"
from typing import assert_type
import bpy.ops.my_blender_addon

assert_type(bpy.ops.my_blender_addon.build(), bpy.ops.my_blender_addon.OperatorProps)
"#,
        file!(),
        line!(),
    )
}

#[test]
fn test_injectable_stub_merges_addon_data_module_without_bound_name_collision() -> anyhow::Result<()>
{
    let mut env = TestEnv::new();
    env.add_with_path("BlenderExampleAddon1", "BlenderExampleAddon1/__init__.py", "");
    env.add_with_path(
        "BlenderExampleAddon1.groups",
        "BlenderExampleAddon1/groups.py",
        r#"
def update_lights_for_mod(_x) -> None:
    pass
"#,
    );
    env.add_with_path(
        "bpy",
        "bpy/__init__.pyi",
        r#"
class types:
    class Context: ...
    class Light:
        color: object
    class PropertyGroup: ...

class props:
    @staticmethod
    def StringProperty(**kwargs): ...
    @staticmethod
    def FloatProperty(**kwargs): ...
"#,
    );
    env.add_with_path(
        "mathutils",
        "mathutils/__init__.pyi",
        r#"
class Color: ...
"#,
    );
    env.add_with_path(
        "BlenderExampleAddon1.data",
        "BlenderExampleAddon1/data.py",
        r#"
import bpy
import mathutils


def _update_lights_for_mod(self: 'MyPropertyGroup2', context: bpy.types.Context) -> None:
    from . import groups

    groups.update_lights_for_mod(self)


def get_color_callback(self: bpy.types.Light) -> object:
    return self.color


class MyPropertyGroup1(bpy.types.PropertyGroup):
    object_name: bpy.props.StringProperty()
    mod_name: bpy.props.StringProperty()
    value_float: bpy.props.FloatProperty(unit='POWER', precision=5)


class MyPropertyGroup2(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(update=_update_lights_for_mod)
    type: bpy.props.StringProperty(update=_update_lights_for_mod)
    factor: bpy.props.FloatProperty(update=_update_lights_for_mod)

    def init(self, mod_type: str):
        self.type = mod_type
        self.name = mod_type
"#,
    );

    let temp = tempdir()?;
    let root = temp.path();
    fs::create_dir_all(root.join(".injectable_stubs/BlenderExampleAddon1"))?;
    fs::write(
        root.join(".injectable_stubs/BlenderExampleAddon1/data.pyi"),
        r#"
import bpy

class MyPropertyGroup1(bpy.types.PropertyGroup):
    object_name: str
    mod_name: str
    value_float: float
    value_float_preview: float

class MyPropertyGroup2(bpy.types.PropertyGroup):
    name: str
    type: str
    factor: float
"#,
    )?;

    testcase_for_macro(
        env.with_config_source_root(root.to_path_buf()),
        r#"
from typing import assert_type
from BlenderExampleAddon1.data import MyPropertyGroup1, MyPropertyGroup2

assert_type(MyPropertyGroup1.object_name, str)
assert_type(MyPropertyGroup1.value_float_preview, float)
assert_type(MyPropertyGroup2.name, str)
"#,
        file!(),
        line!(),
    )
}

#[test]
fn test_injectable_stub_does_not_panic_on_bound_name_range_collision() -> anyhow::Result<()> {
    let mut env = TestEnv::new();
    env.add_with_path(
        "foo",
        "foo.py",
        r#"
T = int
yyyyy = T

class C:
    pass
"#,
    );

    let temp = tempdir()?;
    let root = temp.path();
    fs::create_dir_all(root.join(".injectable_stubs"))?;
    fs::write(
        root.join(".injectable_stubs/foo.pyi"),
        r#"
class C:
    x: T
"#,
    )?;

    testcase_for_macro(
        env.with_config_source_root(root.to_path_buf()),
        r#"
import foo
foo.C.x
"#,
        file!(),
        line!(),
    )
}
