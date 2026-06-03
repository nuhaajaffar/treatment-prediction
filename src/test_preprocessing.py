from preprocessing import (
    load_data,
    prepare_pcr_data,
    prepare_rfs_data,
    prepare_test_data,
    build_preprocessing_pipeline
)

TRAIN_PATH = "data/TrainDataset2025.xls"
TEST_PATH = "data/TestDatasetExample.xls"

def main():
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    X_pcr, y_pcr = prepare_pcr_data(train_df)
    X_rfs, y_rfs = prepare_rfs_data(train_df)
    test_ids, X_test = prepare_test_data(test_df)

    preprocessor = build_preprocessing_pipeline()

    X_pcr_processed = preprocessor.fit_transform(X_pcr)

    print("PCR features shape:", X_pcr.shape)
    print("PCR target shape:", y_pcr.shape)
    print("RFS features shape:", X_rfs.shape)
    print("RFS target shape:", y_rfs.shape)
    print("Test features shape:", X_test.shape)
    print("Test IDs:", test_ids.tolist())
    print("Processed PCR shape:", X_pcr_processed.shape)

if __name__ == "__main__":
    main()