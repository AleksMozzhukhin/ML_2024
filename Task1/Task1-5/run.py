

from json import load, dumps
from glob import glob
from os import environ
from os.path import join
from sys import argv, exit


def run_single_test(data_dir, output_dir):
    from pytest import main
    exit(main(['-vv', '-p', 'no:cacheprovider', join(data_dir, 'test.py')]))


def check_test(data_dir):
    pass


def grade(data_path):
    results = load(open(join(data_path, 'results.json')))
    max_mark = 5
    grade_mapping = [1, 1, 1, 1, 1]
    total_grade = 0
    ok_count = 0
    for result, grade in zip(results, grade_mapping):
        if result['status'] == 'Ok':
            total_grade += grade
            ok_count += 1
    total_count = len(results)
    description = '%02d/%02d' % (ok_count, total_count)
    mark = total_grade / sum(grade_mapping) * max_mark
    res = {'description': description, 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print('Usage: %s mode data_dir output_dir' % argv[0])
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            check_test(data_dir)
        elif mode == 'grade':
            grade(data_dir)
    else:
        # Script is running locally
        if len(argv) != 3:
            print(f'Usage: {argv[0]} test/unittest test_name')
            exit(0)

        mode = argv[1]
        test_name = argv[2]
        test_dir = glob(f'python_intro_public_test/[0-9][0-9]_{mode}_{test_name}_input')
        if not test_dir:
            print('Test not found')
            exit(0)

        from pytest import main
        exit(main(['-vv', join(test_dir[0], 'test.py')]))
