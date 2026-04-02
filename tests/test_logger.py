import csv
from pathlib import Path
from drone_rl.utils.logger import TrainingLogger


def test_csv_written_with_episode_and_step(tmp_path):
    logger = TrainingLogger(
        run_name="test_run",
        output_dir=str(tmp_path),
        use_wandb=False,
    )
    logger.log(episode=1, step=10, metrics={"episode_reward": 5.0, "coverage_fraction": 0.3})
    logger.log(episode=2, step=25, metrics={"episode_reward": 8.0, "coverage_fraction": 0.5})
    logger.close()

    csv_path = tmp_path / "train_log.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 2
    assert rows[0]["episode"] == "1"
    assert rows[0]["episode_reward"] == "5.0"
    assert rows[1]["coverage_fraction"] == "0.5"


def test_logger_skips_wandb_when_disabled(tmp_path):
    # Should not raise even if wandb is not configured
    logger = TrainingLogger(run_name="nowandb", output_dir=str(tmp_path), use_wandb=False)
    logger.log(episode=1, step=1, metrics={"loss": 0.1})
    logger.close()


def test_logger_skips_wandb_gracefully_when_enabled_but_unconfigured(tmp_path):
    # use_wandb=True but no API key — must not raise
    logger = TrainingLogger(run_name="nowandb", output_dir=str(tmp_path), use_wandb=True)
    logger.log(episode=1, step=1, metrics={"loss": 0.1})
    logger.close()
