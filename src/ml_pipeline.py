from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def treinar_modelo(df):
    df = df.dropna(subset=["hora", "causa_acidente"])
    df["risco"] = (df["feridos_leves"] > 0).astype(int)

    features = ["hora", "causa_acidente", "tipo_acidente", "condicao_metereologica"]
    df = df[features + ["risco"]].dropna()

    for col in features:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    X = df[features]
    y = df["risco"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBClassifier()
    model.fit(X_train, y_train)

    return model
