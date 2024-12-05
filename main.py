from src.image_processing import load_data
from src.feature_extraction import load_base_model, extract_features
from src.model_training import train_models
from src.ensemble_model import create_ensemble, evaluate_ensemble

# Paths
train_dir = r"C:\Users\Shree\SML_DL_PROJECT\SML_DL_PROJECT\Dataset\Dataset\train"
test_dir = r"C:\Users\Shree\SML_DL_PROJECT\SML_DL_PROJECT\Dataset\Dataset\test"
feature_dir = r"C:\Users\Shree\SML_DL_PROJECT\SML_DL_PROJECT\features"
model_dir = r"C:\Users\Shree\SML_DL_PROJECT\SML_DL_PROJECT\Models"

# Load Data
train_data, test_data = load_data(train_dir, test_dir)

# Load Models
resnet = load_base_model('resnet')
vggnet = load_base_model('vggnet')
xception = load_base_model('xception')

# Extract Features
resnet_features, labels = extract_features(resnet, train_data, feature_dir, "resnet")
vggnet_features, _ = extract_features(vggnet, train_data, feature_dir, "vggnet")
xception_features, _ = extract_features(xception, train_data, feature_dir, "xception")

# Combine Features
combined_features = np.concatenate([resnet_features, vggnet_features, xception_features], axis=1)

# Train Models
svm, dt, knn, nb, lr = train_models(combined_features, labels, model_dir)

# Ensemble
ensemble = create_ensemble([('svm', svm), ('dt', dt), ('knn', knn), ('nb', nb), ('lr', lr)])
ensemble.fit(combined_features, labels)

# Evaluate
X_test = combined_features  # Replace with actual test features
y_test = labels             # Replace with actual test labels
evaluate_ensemble(ensemble, X_test, y_test)
