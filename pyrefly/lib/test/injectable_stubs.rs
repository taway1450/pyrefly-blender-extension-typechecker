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
