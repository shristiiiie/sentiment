import os
import pandas as pd
import string
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords once
try:
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')

def clean_text(text):
    """Clean and preprocess text data."""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    print(f"Attempting to load data from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        print("Current working directory:", os.getcwd())
        print("Files in current directory:", os.listdir('.'))
        
        # Try to find the file in common locations
        possible_paths = [
            'data/filtered_dataset_expanded.csv',
            'ai_model/data/filtered_dataset_expanded.csv',
            '../data/filtered_dataset_expanded.csv',
            '../../data/filtered_dataset_expanded.csv',
            './filtered_dataset_expanded.csv'
        ]
        
        print("\nSearching for file in possible locations:")
        for path in possible_paths:
            if os.path.exists(path):
                print(f"FOUND: {path}")
                file_path = path
                break
            else:
                print(f"NOT FOUND: {path}")
        else:
            raise FileNotFoundError(f"Could not find filtered_dataset_expanded.csv in any expected location")
    
    data = pd.read_csv(file_path)
    print(f"Successfully loaded data with shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Clean column names
    data.columns = data.columns.str.strip()
    
    # Check for required columns
    if 'Text' not in data.columns or 'Sentiment' not in data.columns:
        print("Available columns:", list(data.columns))
        raise ValueError("Required columns 'Text' and 'Sentiment' not found in dataset")
    
    # Drop rows with missing values
    initial_shape = data.shape[0]
    data = data.dropna(subset=['Text', 'Sentiment'])
    print(f"Dropped {initial_shape - data.shape[0]} rows with missing values")
    
    # Clean text
    data['Text'] = data['Text'].apply(clean_text)
    
    return data

def balance_data(X, y):
    """Balance the dataset by downsampling the majority class."""
    df = pd.concat([X, y], axis=1)
    
    class_counts = df['Sentiment'].value_counts()
    print("Original class distribution:")
    print(class_counts)
    
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    majority_df = df[df['Sentiment'] == majority_class]
    minority_df = df[df['Sentiment'] == minority_class]
    
    # Downsample majority class
    majority_downsampled = resample(
        majority_df,
        replace=False,
        n_samples=len(minority_df),
        random_state=42
    )
    
    # Combine balanced data
    df_balanced = pd.concat([minority_df, majority_downsampled])
    
    return df_balanced['Text'], df_balanced['Sentiment']

def find_project_root():
    """Find the project root directory containing the data folder."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Start from current directory and go up to find data folder
    search_dir = current_dir
    for _ in range(5):  # Search up to 5 levels up
        data_dir = os.path.join(search_dir, 'data')
        if os.path.exists(data_dir):
            return search_dir
        search_dir = os.path.dirname(search_dir)
    
    # If not found, return current directory
    return current_dir

def main():
    print("Starting model training...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    
    # Try multiple approaches to find the correct paths
    project_root = find_project_root()
    print(f"Project root detected as: {project_root}")
    
    # Try different possible data paths
    possible_data_paths = [
        os.path.join(project_root, 'data', 'filtered_dataset_expanded.csv'),
        os.path.join(project_root, 'ai_model', 'data', 'filtered_dataset_expanded.csv'),
        'data/filtered_dataset_expanded.csv',
        'ai_model/data/filtered_dataset_expanded.csv',
        './filtered_dataset_expanded.csv'
    ]
    
    data_path = None
    for path in possible_data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("ERROR: Could not find the data file!")
        print("Please ensure 'filtered_dataset_expanded.csv' exists in one of these locations:")
        for path in possible_data_paths:
            print(f"  - {path}")
        return
    
    print(f"Using data file: {data_path}")
    
    # Set up model directory
    model_dir = os.path.join(project_root, 'ai_model', 'model')
    if not os.path.exists(model_dir):
        # Try alternative model directory location
        model_dir = os.path.join(project_root, 'model')
    
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model will be saved to: {model_dir}")
    
    try:
        # Load and preprocess data
        data = load_and_preprocess_data(data_path)
        
        # Balance the data
        X, y = balance_data(data['Text'], data['Sentiment'])
        print("Class distribution after balancing:")
        print(y.value_counts())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Vectorize the text
        print("Creating TF-IDF vectors...")
        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=10000  # Limit features to prevent memory issues
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_vec.shape}")
        
        # Train the model
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,  # Reduced for faster training
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train_vec, y_train)
        
        # Save the model and vectorizer
        model_path = os.path.join(model_dir, 'model.pkl')
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
        
        # Evaluate the model
        print("Evaluating model...")
        y_pred = model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTraining complete!")
        print(f"Test accuracy: {accuracy:.4f}")
        print("\nDetailed classification report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance (top 10)
        feature_names = vectorizer.get_feature_names_out()
        importances = model.feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        
        print("\nTop 10 most important features:")
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")
            
    except Exception as e:
        print(f"ERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()
