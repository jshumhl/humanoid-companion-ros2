"""Which .env a run loads."""

from voice_agent.env import find_dotenv_for


def test_the_env_beside_the_config_wins_over_one_in_the_working_directory(tmp_path):
    """A repo-root .env may belong to another program; the deployment's own is meant."""
    (tmp_path / ".env").write_text("LLM_API_KEY=other-program\n")
    deployment = tmp_path / "deployment" / "voice_agent"
    deployment.mkdir(parents=True)
    (deployment / ".env").write_text("DASHSCOPE_API_KEY=ours\n")

    assert find_dotenv_for(deployment / "config.yaml", cwd=tmp_path) == deployment / ".env"


def test_a_repo_root_env_is_found_by_walking_up_from_the_config(tmp_path):
    (tmp_path / ".env").write_text("DASHSCOPE_API_KEY=root\n")
    module = tmp_path / "src" / "voice_agent"
    module.mkdir(parents=True)

    assert find_dotenv_for(module / "config.yaml", cwd="/") == tmp_path / ".env"


def test_the_working_directory_is_the_fallback(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / ".env").write_text("DASHSCOPE_API_KEY=cwd\n")

    # Nothing beside the config or above it, so the working directory's is used.
    assert find_dotenv_for(config_dir / "config.yaml", cwd=workdir) == workdir / ".env"
