import psycopg2
from config import host, database, user, password
import pandas as pd
df = pd.read_csv('credit_score.csv', encoding='latin-1')



columns_to_insert = ['CUST_ID', 'INCOME', 'SAVINGS', 'DEBT', 'R_SAVINGS_INCOME', 'R_DEBT_INCOME',
                'R_DEBT_SAVINGS', 'T_CLOTHING_12', 'T_CLOTHING_6', 'R_CLOTHING', 'R_CLOTHING_INCOME',
                'R_CLOTHING_SAVINGS', 'R_CLOTHING_DEBT', 'T_EDUCATION_12', 'T_EDUCATION_6', 'R_EDUCATION',
                'CAT_DEBT', 'CAT_CREDIT_CARD', 'CAT_MORTGAGE', 'CAT_SAVINGS_ACCOUNT', 'CAT_DEPENDENTS', 'CREDIT_SCORE']

try:
    conn = psycopg2.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    conn.autocommit = True

    with conn.cursor() as cursor:

        cursor.execute(
            "SELECT EXISTS("
            "SELECT * FROM information_schema.tables WHERE table_name='credit_score_new');")
        table_exists = cursor.fetchone()[0]

        if not table_exists:
            cursor.execute(
                '''CREATE TABLE credit_score_new (
    CUST_ID VARCHAR(255)PRIMARY KEY ,
    INCOME FLOAT,
    SAVINGS FLOAT,
    DEBT FLOAT,
    R_SAVINGS_INCOME FLOAT,
    R_DEBT_INCOME FLOAT,
    R_DEBT_SAVINGS FLOAT,
    T_CLOTHING_12 FLOAT,
    T_CLOTHING_6 FLOAT,
    R_CLOTHING FLOAT,
    R_CLOTHING_INCOME FLOAT,
    R_CLOTHING_SAVINGS FLOAT,
    R_CLOTHING_DEBT FLOAT,
    T_EDUCATION_12 FLOAT,
    T_EDUCATION_6 FLOAT,
    R_EDUCATION FLOAT,
    CAT_DEBT VARCHAR(255),
    CAT_CREDIT_CARD VARCHAR(255),
    CAT_MORTGAGE VARCHAR(255),
    CAT_SAVINGS_ACCOUNT VARCHAR(255),
    CAT_DEPENDENTS INTEGER,
    CREDIT_SCORE INTEGER
);'''
            )
        print("Table created successfully")
        end = 0
    if end != 0 :
        with conn.cursor() as cursor:
            for index, row in df.iterrows():
                values = [row[col] for col in columns_to_insert]
                placeholders = ', '.join(['%s'] * len(columns_to_insert))
                query = '''
                   INSERT INTO credit_score_new ({})
                   VALUES ({})
                '''.format(', '.join(columns_to_insert), placeholders)
                cursor.execute(query, values)
                print("Inserted")

    with conn.cursor() as cursor:
        cursor.execute("SELECT INCOME, SAVINGS, DEBT, CREDIT_SCORE FROM credit_score_new;")
        results = cursor.fetchall()
        db_data = pd.DataFrame(results, columns=["INCOME", "SAVINGS", "DEBT", "CREDIT_SCORE"])
        income = db_data["INCOME"]
        savings = db_data["SAVINGS"]
        debt = db_data["DEBT"]
        credit_score = db_data["CREDIT_SCORE"]

    print("data fetched successfully")
except Exception as e:
    print('Error occurred:', e)
finally:

    print('Connection closed')




from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import pickle

X = db_data[["INCOME", "SAVINGS", "DEBT"]]

Y = db_data["CREDIT_SCORE"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)



rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)

gb_predictions = gb_model.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_predictions)

rf_score = cross_val_score(rf_model, X_test, y_test, cv=5).mean()
gb_score = cross_val_score(gb_model, X_test, y_test, cv=5).mean()


if rf_score > gb_score:
    best_model = rf_model
else:
    best_model = gb_model

lgb_predictions = best_model.predict(X_test)


pickle.dump(best_model, open("model.pkl", "wb"))


