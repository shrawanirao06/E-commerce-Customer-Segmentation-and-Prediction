# Data Dictionary — `data/raw/data.csv`

This is the UCI "Online Retail" transaction log (UK-based online gift
retailer, Dec 2010 – Dec 2011, 541,909 rows, 4,372 distinct customers).

| Column        | Description                                             |
|---------------|----------------------------------------------------------|
| `InvoiceNo`   | Unique identifier for each transaction/invoice. Invoice numbers starting with "C" are cancellations. |
| `StockCode`   | Product code.                                             |
| `Description` | Product name.                                              |
| `Quantity`    | Number of units purchased (negative for cancellations/returns). |
| `InvoiceDate` | Date and time of the transaction.                          |
| `UnitPrice`   | Price per unit of the product.                              |
| `CustomerID`  | Unique identifier for each customer. ~25% of rows have no CustomerID and are dropped during cleaning (guest/unattributed orders). |
| `Country`     | Country where the transaction occurred.                    |

Derived during feature engineering (see `src/ecommerce_segmentation/`):

| Column               | Description |
|----------------------|-------------|
| `Revenue`             | `Quantity * UnitPrice`. |
| `Recency`             | Days since the customer's most recent purchase, relative to a reference date. |
| `Frequency`           | Number of distinct invoices for the customer. |
| `Monetary`            | Total revenue from the customer. |
| `AverageOrderValue`   | `Monetary / Frequency` (churn model only). |
