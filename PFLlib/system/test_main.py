import subprocess


def test_main():
    result = subprocess.run(
        [
            "python",
            "main.py",
            "-data",
            "MNIST",
            "-m",
            "cnn",
            "-algo",
            "FedAvg",
            "-gr",
            "5",
        ],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"main.py failed with return code {result.returncode}"

    assert (
        "Expected output" in result.stdout
    ), "main.py did not produce expected output"
