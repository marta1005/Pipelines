
import pandas as pd


def normalizer_transformer(df: pd.DataFrame, params: dict ):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    from sklearn.compose import ColumnTransformer
    
    inputs = params['inputs']
    outputs = params['outputs']


    if isinstance(outputs, str):
        outputs = list(map(str.strip, outputs.strip('[]').split(',')))

    ## masking data for each output
    df_mask = ~df[outputs].apply(pd.isnull).any(axis=1)
    columns = list(set([*inputs, *outputs]))
    train = df.loc[df_mask][columns]       
    Train_X = train[inputs]


    categorical_features = []
    numeric_features = []

    print("Output:", outputs)
    if ('categorical_features' in params.keys()):  ## TODO: analyze input_table instead
        categorical_features = list(map(str.strip, params['categorical_features'].strip('[]').split(',')))
        # categorical_features = eval(flow_variables['metadata', 'Feature_selection', 'preprocess', 'categorical_features'])
    else:
        if (df[inputs].dtypes == object).any() or (df[inputs].dtypes == int).any():
            categorical_features = Train_X.select_dtypes([object, int]).columns.tolist()

    if ('numerical_features' in params.keys()):  ## TODO: analyze input_table instead
        numeric_features = list(map(str.strip, params['numerical_features'].strip('[]').split(',')))
        # numeric_features = eval(flow_variables['metadata', 'Feature_selection', 'preprocess', "numerical_variables"])
    else:
        if (df[inputs].dtypes == float).any():
            numeric_features = df[inputs].select_dtypes([float]).columns.tolist()



    print("Number of Categorical Variables:", len(categorical_features), categorical_features)
    print("Number of Numerical Variables:", len(numeric_features),numeric_features )


    if len(categorical_features) and len(numeric_features):
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        numeric_transformer = StandardScaler()
        PostprocessTransformer = ColumnTransformer(
                                                transformers=[
                                                    ("num", numeric_transformer, numeric_features),
                                                    ("cat", categorical_transformer, categorical_features),
                                                    ]
                                                )
    else:
        PostprocessTransformer = StandardScaler()

    transformer_fit = PostprocessTransformer.fit(Train_X)

    return transformer_fit