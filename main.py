from src.preprocess import preprocess

def main():
    # Preprocess the data into training and testing
    X_train, X_test, y_train, y_test = preprocess()

if __name__ == "__main__":
    main()
