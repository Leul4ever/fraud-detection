import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating sample data in {raw_dir}...")
    
    # 1. Fraud_Data.csv
    # user_id,signup_time,purchase_time,purchase_value,device_id,source,browser,sex,age,ip_address,class
    n_samples = 1000
    fraud_data = pd.DataFrame({
        'user_id': range(1, n_samples + 1),
        'signup_time': [pd.to_datetime('2015-01-01 10:00:00') + pd.Timedelta(hours=i) for i in range(n_samples)],
        'purchase_time': [pd.to_datetime('2015-01-01 10:00:01') + pd.Timedelta(hours=i) + pd.Timedelta(seconds=np.random.randint(1, 100000)) for i in range(n_samples)],
        'purchase_value': np.random.randint(10, 100, n_samples),
        'device_id': ['DEVICE' + str(i % 50) for i in range(n_samples)],
        'source': (['SEO', 'Ads', 'Direct'] * (n_samples // 3 + 1))[:n_samples],
        'browser': (['Chrome', 'Firefox', 'IE', 'Safari'] * (n_samples // 4 + 1))[:n_samples],
        'sex': (['M', 'F'] * (n_samples // 2))[:n_samples],
        'age': np.random.randint(18, 70, n_samples),
        'ip_address': np.random.randint(1e8, 4e9, size=n_samples, dtype=np.int64),
        'class': ([0] * int(n_samples * 0.9) + [1] * int(n_samples * 0.1))
    })
    fraud_data.to_csv(raw_dir / "Fraud_Data.csv", index=False)
    print("✓ Created Fraud_Data.csv")
    
    # 2. IpAddress_to_Country.csv
    # lower_bound_ip_address,upper_bound_ip_address,country
    ip_data = pd.DataFrame({
        'lower_bound_ip_address': [0, 1e9, 2e9, 3e9],
        'upper_bound_ip_address': [1e9-1, 2e9-1, 3e9-1, 4e9-1],
        'country': ['United States', 'China', 'Japan', 'Other']
    })
    ip_data.to_csv(raw_dir / "IpAddress_to_Country.csv", index=False)
    print("✓ Created IpAddress_to_Country.csv")
    
    # 3. creditcard.csv
    # Time, V1-V28, Amount, Class
    v_cols = {f'V{i}': np.random.normal(0, 1, n_samples) for i in range(1, 29)}
    credit_data = pd.DataFrame({
        'Time': range(n_samples),
        **v_cols,
        'Amount': np.random.uniform(0, 500, n_samples),
        'Class': ([0] * int(n_samples * 0.99) + [1] * int(n_samples * 0.01))
    })
    credit_data.to_csv(raw_dir / "creditcard.csv", index=False)
    print("✓ Created creditcard.csv")

if __name__ == "__main__":
    create_sample_data()
