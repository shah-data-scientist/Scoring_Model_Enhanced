import sqlite3

print('='*80)
print('DATABASE 1: ./mlflow.db (ROOT)')
print('='*80)
conn1 = sqlite3.connect('mlflow.db')
cursor1 = conn1.cursor()
cursor1.execute('SELECT COUNT(*) as count FROM runs')
count1 = cursor1.fetchone()[0]
cursor1.execute('''SELECT r.name FROM runs r JOIN experiments e ON r.experiment_id = e.experiment_id 
                   WHERE e.name = 'credit_scoring_final_delivery' ORDER BY r.start_time DESC LIMIT 1''')
latest1 = cursor1.fetchone()
print(f'Total runs: {count1}')
if latest1:
    print(f'Latest production run: {latest1[0]}')
else:
    print('Latest production run: NOT FOUND')
conn1.close()

print()
print('='*80)
print('DATABASE 2: ./mlruns/mlflow.db (MLRUNS)')
print('='*80)
conn2 = sqlite3.connect('mlruns/mlflow.db')
cursor2 = conn2.cursor()
cursor2.execute('SELECT COUNT(*) as count FROM runs')
count2 = cursor2.fetchone()[0]
cursor2.execute('''SELECT r.name FROM runs r JOIN experiments e ON r.experiment_id = e.experiment_id 
                   WHERE e.name = 'credit_scoring_final_delivery' ORDER BY r.start_time DESC LIMIT 1''')
latest2 = cursor2.fetchone()
print(f'Total runs: {count2}')
if latest2:
    print(f'Latest production run: {latest2[0]}')
else:
    print('Latest production run: NOT FOUND')
conn2.close()
