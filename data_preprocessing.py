import numpy as np
def preprocess_data(df):
    X = df.drop('label', axis=1).copy()
    y = df['label']

    x_cols = list(range(0, X.shape[1], 3))
    y_cols = [c + 1 for c in x_cols]
    z_cols = [c + 2 for c in x_cols]

    #subract by the wrist coordinates
    X.iloc[:, x_cols] = X.iloc[:, x_cols].sub(X.iloc[:, 0], axis=0)
    X.iloc[:, y_cols] = X.iloc[:, y_cols].sub(X.iloc[:, 1], axis=0)
    X.iloc[:, z_cols] = X.iloc[:, z_cols].sub(X.iloc[:, 2], axis=0)

    #normalize the coordinates by the distance between the wrist and the middle finger
    div=np.sqrt(X['x12']**2+X['y12']**2+X['z12']**2)
    X.iloc[:, x_cols] = X.iloc[:, x_cols].div(div, axis=0)
    X.iloc[:, y_cols] = X.iloc[:, y_cols].div(div, axis=0)
    X.iloc[:, z_cols] = X.iloc[:, z_cols].div(div, axis=0)
    return X, y