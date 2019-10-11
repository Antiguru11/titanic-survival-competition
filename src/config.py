import os

base_path = 'd:\\Projects\\ml\\titanic-survival-competition\\'

input_path = os.path.join(base_path, 'input')
submissions_path = os.path.join(base_path, 'submissions')
tmp_path = os.path.join(base_path, 'temp')

task_type = 0
target_col = 'Survived'
target_vars = {'Survived': 1, 'Died': 0}

index_col = 'PassengerId'
