from aliensilicon.cli import main
from aliensilicon.tasks import BASELINES


def test_judge_exit_codes(tmp_path, capsys):
    assert main(["judge", str(BASELINES / "softmax_naive.py"), "--task", "w2"]) == 0
    assert "CORRECT" in capsys.readouterr().out
    bad = tmp_path / "bad.py"
    bad.write_text((BASELINES / "softmax_naive.py").read_text().replace("    x = sub(x, m)\n", ""))
    assert main(["judge", str(bad), "--task", "w2"]) == 1
    assert "S3" in capsys.readouterr().out


def test_tasks_lists_all(capsys):
    assert main(["tasks"]) == 0
    out = capsys.readouterr().out
    assert all(t in out for t in ("w1", "w2", "w3"))
