import kagglehub


def main() -> str:
    path = kagglehub.dataset_download("patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
    print(f"Dataset available at: {path}")
    return path


if __name__ == "__main__":
    main()
