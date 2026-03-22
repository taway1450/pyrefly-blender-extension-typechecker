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
