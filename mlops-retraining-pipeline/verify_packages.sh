#!/bin/bash
source /home/mlops/mlops-venv/bin/activate
python -c "import airflow, sklearn, pandas, numpy; print('airflow:', airflow.__version__); print('sklearn:', sklearn.__version__); print('pandas:', pandas.__version__); print('numpy:', numpy.__version__)"
