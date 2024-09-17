# Anonymization Algorithm

This project provides an algorithm for data anonymization, specifically for SQL databases. The algorithm is designed to balance GDPR compliance with the need to maintain data utility for various services.

## Key Features

1. **Complex Queries and Flexible Column Selection**
   - **Execute Complex Queries**: Allows executing complex queries on masked data while preserving important relationships.
   - **Flexible Column Selection**: Flexibly select sensitive columns that need to be anonymized.

2. **Generate Valuable Data**
   - **Different Techniques**: Apply various anonymization techniques to sensitive data while ensuring that duplicate values remain consistent.
   - **Consistent Data**: Ensures that duplicate records receive the same anonymized values, maintaining data integrity.

3. **Maintaining Relationships**
   - **Same-Table Storage**: Anonymized data can be stored in the same table as the original data (optional), making it easier to manage and access.
   - **Consistent Join Columns**: The algorithm ensures that columns used in joins are anonymized with consistent values, allowing you to retrieve original data relationships seamlessly.
