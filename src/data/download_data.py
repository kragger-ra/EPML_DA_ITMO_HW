"""Download Customer Churn dataset."""

import urllib.request
from pathlib import Path


def download_churn_data() -> Path:
    """Download customer churn dataset from IBM GitHub repository."""
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

    data_dir = Path(__file__).parents[2] / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_file = data_dir / "customer_churn.csv"

    if output_file.exists():
        print(f"Dataset already exists at {output_file}")
        return output_file

    print(f"Downloading dataset to {output_file}...")
    urllib.request.urlretrieve(url, output_file)  # nosec B310

    print("Dataset downloaded successfully!")
    print(f"File size: {output_file.stat().st_size} bytes")

    return output_file


if __name__ == "__main__":
    download_churn_data()
