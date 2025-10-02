import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

df = pd.read_csv("Auto.csv")


X = df.drop(columns=['mpg', 'name', 'origin'])
y = df['mpg']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


alphas = np.logspace(-3, 3, 50)

ridge_scores = []
lasso_scores = []


for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_scores.append(r2_score(y_test, ridge_pred))

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_scores.append(r2_score(y_test, lasso_pred))


plt.figure(figsize=(8, 5))
plt.plot(alphas, ridge_scores, label='Ridge Regression', marker='o')
plt.plot(alphas, lasso_scores, label='LASSO Regression', marker='x')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.title('R2 Score vs Alpha for Ridge and LASSO Regression')
plt.legend()
plt.grid(True)
plt.show()


best_alpha_ridge = alphas[np.argmax(ridge_scores)]
best_r2_ridge = max(ridge_scores)

best_alpha_lasso = alphas[np.argmax(lasso_scores)]
best_r2_lasso = max(lasso_scores)

print(f"Best Ridge alpha: {best_alpha_ridge:.4f} with R2: {best_r2_ridge:.3f}")
print(f"Best LASSO alpha: {best_alpha_lasso:.4f} with R2: {best_r2_lasso:.3f}")




