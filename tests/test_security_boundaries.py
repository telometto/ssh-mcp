"""Regressions for transport argument boundaries and hostile remote output."""
from __future__ import annotations

import io
import json
import os
import shlex
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssh_mcp import ssh
from ssh_mcp.server import McpServer
from tests.helpers import create_fake_rsync, create_fake_scp, create_fake_ssh


class TransportBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.service = ssh.SshToolService(
            ssh_binary=str(create_fake_ssh(self.root)),
            scp_binary=str(create_fake_scp(self.root)),
            rsync_binary=str(create_fake_rsync(self.root)),
            state_dir=self.root / "state",
        )
        self.addCleanup(self.service.close)

    def _connection_calls(self):
        return (
            (self.service.ssh_exec, {"command": "true"}),
            (self.service.ssh_scp, {"direction": "upload", "sources": ["unused"], "destination": "/tmp"}),
            (self.service.ssh_sync, {"direction": "download", "source": "unused", "destination": "/tmp"}),
            (self.service.ssh_view, {"path": "unused"}),
            (self.service.ssh_create, {"path": "unused", "content": "hello"}),
            (self.service.ssh_edit, {"path": "unused", "old_str": "x", "new_str": "y"}),
            (self.service.ssh_grep, {"pattern": "x"}),
            (self.service.ssh_glob, {"pattern": "*"}),
            (self.service.ssh_start_session, {"observer_mode": "transcript"}),
            (self.service.ssh_ensure_session, {"observer_mode": "transcript"}),
            (self.service.ssh_forward, {"local_port": 12345, "remote_host": "localhost", "remote_port": 80}),
        )

    def test_unsafe_extras_rejected_before_any_transport_on_all_entry_points(self) -> None:
        payloads = [
            ["-o", "ProxyCommand echo marker"], ["-oProxyCommand=echo marker"],
            ["-o", "pRoXyCoMmAnD\techo marker"], ["-voProxyCommand=echo marker"],
            ["-o", "KnownHostsCommand=echo marker"], ["-oKnownHostsCommand echo marker"],
            ["-o", "LocalCommand=echo marker"], ["-o", "PermitLocalCommand=yes"],
            ["-o", "PKCS11Provider=/tmp/provider"], ["-o", "SecurityKeyProvider=/tmp/provider"],
            ["-F", "/tmp/caller-config"], ["-F/tmp/caller-config"],
            ["-o", "Include=/tmp/config"], ["-o", "Match=exec echo marker"],
            ["-I", "/tmp/provider"], ["-S", "/tmp/transport"], ["-S/tmp/transport"],
            ["-vS/tmp/transport"], ["-D", "/tmp/server"], ["-D/tmp/server"], ["-vD/tmp/server"],
            ["-o", "ClearAllForwardings=no"], ["-o", "EscapeChar=~"],
            ["-o", "EnableEscapeCommandline=yes"], ["-e", "~"], ["-O", "forward"],
            ["--"], ["other-host"], ["-o"], ["-F"], ["-vC"],
            ["-o", "ConnectTimeout=10\nProxyCommand=echo marker"],
        ]
        for flag in ("-L", "-R", "-D", "-W"):
            payloads.extend(([flag, "8080:localhost:80"], [f"{flag}8080:localhost:80"],
                             [f"-v{flag[1:]}8080:localhost:80"]))
        for key in ("LocalForward", "RemoteForward", "DynamicForward"):
            payloads.extend((["-o", f"{key} 8080 localhost:80"], [f"-o{key}=8080 localhost:80"]))
        with mock.patch("ssh_mcp.ssh.subprocess.Popen") as spawn:
            for call, arguments in self._connection_calls():
                for extras in payloads:
                    with self.subTest(tool=call.__name__, extras=extras):
                        with self.assertRaises(ssh.ValidationError):
                            call({"target": "example", **arguments, "extra_ssh_args": extras})
            spawn.assert_not_called()

    def test_unsafe_targets_rejected_on_all_entry_points(self) -> None:
        targets = ("-oProxyCommand=echo marker", "-e/tmp/transport", " --host", "user@-host",
                   "host:path", "ssh://user@host", "host/other", "user@@host", "host\nother")
        with mock.patch("ssh_mcp.ssh.subprocess.Popen") as spawn:
            for call, arguments in self._connection_calls():
                for target in targets:
                    with self.subTest(tool=call.__name__, target=target):
                        with self.assertRaises(ssh.ValidationError):
                            call({**arguments, "target": target})
            spawn.assert_not_called()

    def test_safe_targets_and_ipv6_remote_operands(self) -> None:
        for target in ("alias_name", "host.example.", "user-name@host", "127.0.0.1", "::1",
                       "user@[2001:db8::1]", "user@fe80::1%eth0"):
            with self.subTest(target=target):
                settings = ssh.ConnectionSettings.from_arguments({"target": target})
                argv = settings.build_argv("ssh", "true", tty=False)
                self.assertEqual(argv[-3:], ["--", target, "true"])
        self.assertEqual(ssh._build_remote_path("user@::1", "/tmp/a b"), "user@[::1]:'/tmp/a b'")

    def test_safe_options_are_canonicalized_and_preserve_keepalive_overrides(self) -> None:
        settings = ssh.ConnectionSettings.from_arguments({
            "target": "example", "extra_ssh_args": [
                "-v", "-C", "-o", "serveraliveinterval 60", "-oStrictHostKeyChecking=no",
                "-J", "user@jump:2222,[::1]:22", "-F", "/dev/null",
            ],
        })
        argv = settings.build_argv("ssh", "true", tty=False, keepalive=True)
        self.assertIn("ServerAliveInterval=60", argv)
        self.assertNotIn("ServerAliveInterval=30", argv)
        self.assertIn("ServerAliveCountMax=3", argv)
        for control in ("ClearAllForwardings=yes", "PermitLocalCommand=no", "EscapeChar=none"):
            self.assertLess(argv.index(control), argv.index("-v"))
        self.assertIn("user@jump:2222,[::1]:22", argv)
        self.assertIn("StrictHostKeyChecking=no", argv)

    def test_jump_hosts_cannot_supply_options_or_shell_syntax(self) -> None:
        for value in ("-oProxyCommand=marker", "jump,-F/tmp/config", "host;echo marker", "user@-host",
                      "[::1]:-1", "host:65536", "host:", "host:22:33", "jump,"):
            with self.subTest(value=value), self.assertRaises(ssh.ValidationError):
                ssh.ConnectionSettings.from_arguments({"target": "example", "extra_ssh_args": ["-J", value]})

    def test_scp_has_distinct_flag_semantics(self) -> None:
        settings = ssh.ConnectionSettings.from_arguments({"target": "example", "extra_ssh_args": ["-a"]})
        self.assertIn("-a", settings.build_transport_argv("custom-ssh", port_flag="-p"))
        with self.assertRaises(ssh.ValidationError):
            settings.build_transport_argv("custom-scp", port_flag="-P")

    def test_rsync_transport_overrides_and_dangling_options_are_rejected(self) -> None:
        payloads = (["-e/tmp/transport", "--exclude"], ["-ave/tmp/transport"],
                    ["--rsh=/tmp/transport"], ["--rsh", "/tmp/transport"],
                    ["--rsync-p=/tmp/transport"], ["--r=/tmp/transport"],
                    ["--rsync-path=/tmp/transport"], ["--"], ["other-source"], ["--exclude"])
        with mock.patch("ssh_mcp.ssh.subprocess.Popen") as spawn:
            for extras in payloads:
                with self.subTest(extras=extras), self.assertRaises(ssh.ValidationError):
                    self.service.ssh_sync({"target": "example", "direction": "upload", "source": "unused",
                                           "destination": "/tmp", "extra_rsync_args": extras})
            spawn.assert_not_called()
        self.assertEqual(ssh._extra_rsync_args_argument({
            "extra_rsync_args": ["--itemize-changes", "--bwlimit", "100", "--exclude", "-e-file"],
        }), ["--itemize-changes", "--bwlimit=100", "--exclude=-e-file"])

    def test_downloaded_program_cannot_be_selected_as_scp_transport(self) -> None:
        remote_file = self.root / "remote-program"
        remote_file.write_text("#!/bin/sh\nexit 99\n")
        remote_file.chmod(0o755)
        downloaded = self.root / "downloaded-program"
        result = self.service.ssh_scp({"target": "example", "direction": "download",
                                       "sources": [str(remote_file)], "destination": str(downloaded)})
        self.assertTrue(result["ok"])
        self.assertTrue(downloaded.stat().st_mode & 0o111)
        with mock.patch("ssh_mcp.ssh.subprocess.Popen") as spawn:
            with self.assertRaises(ssh.ValidationError):
                self.service.ssh_scp({"target": "example", "direction": "upload",
                                      "sources": [str(remote_file)], "destination": "/tmp/unused",
                                      "extra_ssh_args": ["-S", str(downloaded)]})
            spawn.assert_not_called()

    def test_dash_and_colon_local_transfer_paths_remain_local(self) -> None:
        source = self.root / "-ehost:source"
        source.write_text("safe local file")
        remote = self.root / "remote"
        remote.mkdir()
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            for tool in ("scp", "sync"):
                with self.subTest(tool=tool):
                    log = self.root / f"{tool}.jsonl"
                    call = self.service.ssh_scp if tool == "scp" else self.service.ssh_sync
                    source_args = {"sources": [source.name]} if tool == "scp" else {"source": source.name}
                    log_variable = "FAKE_SCP_LOG" if tool == "scp" else "FAKE_RSYNC_LOG"
                    with mock.patch.dict(os.environ, {log_variable: str(log)}):
                        uploaded = call({"target": "example", "direction": "upload", **source_args,
                                         "destination": str(remote)})
                        local_destination = f"-download:{tool}"
                        source_args = ({"sources": [str(remote / source.name)]} if tool == "scp"
                                       else {"source": str(remote / source.name)})
                        downloaded = call({"target": "example", "direction": "download", **source_args,
                                           "destination": local_destination})
                    self.assertTrue(uploaded["ok"])
                    self.assertTrue(downloaded["ok"])
                    self.assertEqual(Path(local_destination).read_text(), "safe local file")
                    for record in log.read_text().splitlines():
                        argv = json.loads(record)["argv"]
                        operands = argv[argv.index("--") + 1:]
                        self.assertEqual(len(operands), 2)
                        self.assertTrue(any(operand.startswith("./-") for operand in operands))
                        if tool == "sync":
                            transport = shlex.split(argv[argv.index("-e") + 1])
                            self.assertEqual(transport[0], self.service._configured_ssh_binary)
                            self.assertNotIn("--", transport)
        finally:
            os.chdir(previous)

    def test_forward_flags_are_inserted_before_delimiter_for_aliases_matching_option_values(self) -> None:
        for target in ("none", "30", "3", "yes"):
            with self.subTest(target=target):
                connection = ssh.ConnectionSettings(target=target)
                argv = connection.build_argv("ssh", None, tty=False, keepalive=True,
                                             forwarding_args=["-N", "-L", "127.0.0.1:12345:host:80"])
                self.assertEqual(argv[-5:], ["-N", "-L", "127.0.0.1:12345:host:80", "--", target])
                self.assertNotIn("ClearAllForwardings=yes", argv)
                self.assertIn("EscapeChar=none", argv)


class OutputBudgetTests(unittest.TestCase):
    CAP = 4096

    def setUp(self) -> None:
        patch = mock.patch("ssh_mcp.ssh.DEFAULT_CAPTURE_LIMIT_BYTES", self.CAP)
        patch.start()
        self.addCleanup(patch.stop)

    def _run(self, code, *, runner=ssh._run_without_pty, timeout=3, **kwargs):
        return runner([sys.executable, "-c", code], timeout=timeout, **kwargs)

    def test_all_runners_stop_over_cap_stdout_and_stderr(self) -> None:
        for runner in (ssh._run_without_pty, ssh._run_without_pty_binary_safe, ssh._run_with_pty):
            for fd in (1, 2):
                with self.subTest(runner=runner.__name__, fd=fd):
                    result = self._run(
                        f"import os,time; os.write({fd}, str(os.getpid()).encode()+b'\\n'+b'x'*{self.CAP + 200}); time.sleep(30)",
                        runner=runner,
                    )
                    stream = "stdout" if runner is ssh._run_with_pty or fd == 1 else "stderr"
                    self.assertTrue(result["output_limit_exceeded"])
                    self.assertTrue(result[f"{stream}_truncated"])
                    self.assertEqual(len(result[stream].encode()), self.CAP)
                    self.assertFalse(result["timed_out"])
                    self.assertLess(result["duration_ms"], 2500)
                    pid = int(result[stream].splitlines()[0])
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)

    def test_exact_cap_is_complete_on_both_pipes(self) -> None:
        result = self._run(f"import os; os.write(1,b'a'*{self.CAP}); os.write(2,b'b'*{self.CAP})")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["output_limit_exceeded"])
        self.assertEqual(result["stdout"], "a" * self.CAP)
        self.assertEqual(result["stderr"], "b" * self.CAP)

    def test_cap_applies_without_timeout_and_after_producer_exit(self) -> None:
        result = self._run(f"import os; os.write(1,b'x'*{self.CAP + 1})", timeout=None)
        self.assertTrue(result["stdout_truncated"])
        self.assertEqual(len(result["stdout"]), self.CAP)
        self.assertFalse(result["timed_out"])

    def test_byte_cap_precedes_utf8_decode(self) -> None:
        result = self._run(f"import os; os.write(1, ('é'*{self.CAP}).encode())")
        self.assertTrue(result["output_limit_exceeded"])
        self.assertEqual(len(result["stdout"]), self.CAP // 2)

    def test_large_stdin_is_serviced_concurrently_with_both_outputs(self) -> None:
        content = "input\r\n" * 100_000
        result = self._run(
            f"import os,sys; os.write(1,b'o'*{self.CAP}); os.write(2,b'e'*{self.CAP}); "
            f"data=sys.stdin.buffer.read(); sys.exit(0 if len(data)=={len(content.encode())} else 1)",
            runner=ssh._run_without_pty_binary_safe, input_text=content,
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])
        self.assertFalse(result["output_limit_exceeded"])

    def test_text_and_binary_newline_behavior_is_preserved(self) -> None:
        code = "import os; os.write(1,b'a\\r\\nb\\rc\\n')"
        self.assertEqual(self._run(code)["stdout"], "a\nb\nc\n")
        self.assertEqual(self._run(code, runner=ssh._run_without_pty_binary_safe)["stdout"], "a\r\nb\rc\n")

    def test_timeout_remains_distinct_from_output_limit(self) -> None:
        for runner in (ssh._run_without_pty, ssh._run_without_pty_binary_safe, ssh._run_with_pty):
            with self.subTest(runner=runner.__name__):
                result = self._run("import os,time; os.write(1,b'partial'); time.sleep(30)",
                                   runner=runner, timeout=0.2)
                self.assertTrue(result["timed_out"])
                self.assertFalse(result["output_limit_exceeded"])
                self.assertEqual(result["stdout"], "partial")

    @unittest.skipUnless(Path("/proc/self/status").exists(), "Linux process status is needed to check orphan cleanup")
    def test_descendant_is_stopped_even_after_group_leader_exits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "pid"
            code = (
                "import os,signal,time\n"
                "if os.fork(): os._exit(0)\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                f"open({str(pid_file)!r},'w').write(str(os.getpid()))\n"
                f"os.write(1,b'x'*{self.CAP + 1})\n"
                "time.sleep(30)\n"
            )
            result = self._run(code)
            self.assertTrue(result["output_limit_exceeded"])
            pid = int(pid_file.read_text())
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                status = Path(f"/proc/{pid}/status")
                if not status.exists() or "State:\tZ" in status.read_text():
                    break
                time.sleep(0.01)
            else:
                os.kill(pid, signal.SIGKILL)
                self.fail("output-producing descendant survived the capture limit")


class FileOutputBudgetTests(unittest.TestCase):
    def test_all_file_tools_reject_over_budget_protocol_and_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            producer = root / "producer"
            service = ssh.SshToolService(ssh_binary=str(producer), state_dir=root / "state")
            self.addCleanup(service.close)
            calls = (("ssh_view", {"path": "unused", "force_read_large_files": True}),
                     ("ssh_create", {"path": "unused", "content": "input"}),
                     ("ssh_edit", {"path": "unused", "edits": [{"old_str": "x", "new_str": "y"}]}),
                     ("ssh_grep", {"pattern": "x", "head_limit": 1}),
                     ("ssh_glob", {"pattern": "*", "head_limit": 1}))
            with mock.patch("ssh_mcp.ssh.DEFAULT_CAPTURE_LIMIT_BYTES", 128):
                for fd in (1, 2):
                    producer.write_text(f"#!/usr/bin/env python3\nimport os\nos.write({fd}, b'x'*129)\n")
                    producer.chmod(0o755)
                    for name, arguments in calls:
                        with self.subTest(tool=name, fd=fd):
                            with mock.patch.object(service, "_run_remote_script", wraps=service._run_remote_script) as run:
                                with self.assertRaises(ssh.RemoteFileError) as raised:
                                    getattr(service, name)({"target": "example", **arguments})
                                self.assertEqual(run.call_count, 1)
                            self.assertTrue(raised.exception.details["output_limit_exceeded"])

    def test_failed_forward_diagnostics_use_the_receiver_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            producer = root / "producer"
            producer.write_text("#!/usr/bin/env python3\nimport os\nos.write(2,b'x'*129)\nraise SystemExit(4)\n")
            producer.chmod(0o755)
            service = ssh.SshToolService(ssh_binary=str(producer), state_dir=root / "state")
            self.addCleanup(service.close)
            with mock.patch("ssh_mcp.ssh.DEFAULT_CAPTURE_LIMIT_BYTES", 128):
                result = service.ssh_forward({"target": "none", "direction": "local", "local_port": 12345,
                                              "remote_host": "localhost", "remote_port": 80})
            self.assertFalse(result["running"])
            self.assertTrue(result["output_limit_exceeded"])
            self.assertTrue(result["stderr_truncated"])
            self.assertEqual(len(result["stderr"]), 128)

    def test_file_tools_fail_closed_and_server_remains_responsive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = ssh.SshToolService(ssh_binary=str(create_fake_ssh(root)), state_dir=root / "state")
            self.addCleanup(service.close)
            file = root / "file.txt"
            original = "original line\n" + "body line\n" * 100
            file.write_text(original)
            server = McpServer(stdin=io.BytesIO(), stdout=io.BytesIO(), tool_service=service)
            with mock.patch("ssh_mcp.ssh.DEFAULT_CAPTURE_LIMIT_BYTES", 128):
                result = server._handle_tool_call({"name": "ssh_edit", "arguments": {
                    "target": "example", "path": str(file),
                    "edits": [{"old_str": "original", "new_str": "changed"}],
                }})
                self.assertTrue(result["isError"])
                details = result["structuredContent"]["details"]
                self.assertTrue(details["output_limit_exceeded"])
                self.assertEqual(details["output_limit_bytes"], 128)
                self.assertEqual(file.read_text(), original)
                next_result = server._handle_tool_call({"name": "ssh_exec", "arguments": {
                    "target": "example", "command": "printf next-request",
                }})
                self.assertTrue(next_result["structuredContent"]["ok"])
                self.assertEqual(next_result["structuredContent"]["stdout"], "next-request")

    def test_search_head_limits_cannot_bypass_receiver_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = ssh.SshToolService(ssh_binary=str(create_fake_ssh(root)), state_dir=root / "state")
            self.addCleanup(service.close)
            for index in range(20):
                (root / f"file-{index:02}.txt").write_text("match\n")
            with mock.patch("ssh_mcp.ssh.DEFAULT_CAPTURE_LIMIT_BYTES", 128):
                for name, arguments in (("ssh_glob", {"pattern": "*.txt"}),
                                        ("ssh_grep", {"pattern": "match"})):
                    with self.subTest(tool=name), self.assertRaises(ssh.RemoteFileError) as raised:
                        getattr(service, name)({"target": "example", "path": str(root), "head_limit": 1,
                                                **arguments})
                    self.assertTrue(raised.exception.details["output_limit_exceeded"])


if __name__ == "__main__":
    unittest.main()
