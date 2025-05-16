from typing import List


def hello(name: str = None) -> str:
    if not name:
        return "Hello!"
    else:
        return "Hello, " + name + "!"


def int_to_roman(num: int) -> str:
    pairs_of_numbers = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]
    answer = ""

    for int_num, roman_num in pairs_of_numbers:
        answer += roman_num * (num // int_num)
        num %= int_num

    return answer


def longest_common_prefix(strs_input: List[str]) -> str:
    if not strs_input:
        return ""
    pref = ""
    min_len = min(len(s.lstrip()) for s in strs_input)
    for i in range(min_len):
        prev = strs_input[0].lstrip()[i]
        for j in range(1, len(strs_input)):
            if strs_input[j].lstrip()[i] != prev:
                return pref
        pref += prev
    return pref


def primes() -> int:
    max_prime = 100
    list_numbers = [True for i in range(max_prime + 1)]
    list_numbers[0] = list_numbers[1] = False
    list_primes = []
    current_number = 2

    while True:
        for i in range(current_number, max_prime + 1):
            if list_numbers[i]:
                yield i
                list_primes.append(i)
                for not_primes in range(i * i, max_prime + 1, i):
                    list_numbers[not_primes] = False
        current_number = max_prime
        max_prime *= 2
        list_numbers += [True for j in range(max_prime // 2)]
        for found_prime in list_primes:
            for not_primes in range(max(found_prime * found_prime, (current_number // found_prime + 1) * found_prime),
                                    max_prime + 1, found_prime):
                list_numbers[not_primes] = False
        current_number += 1


class BankCard:
    def __init__(self, total_sum: int = 0, balance_limit: int = None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def __call__(self, sum_spent):
        if self.total_sum >= sum_spent:
            self.total_sum -= sum_spent
            print("You spent", sum_spent, "dollars.")
        else:
            print("Not enough money to spend sum_spent dollars.")
            raise ValueError

    def __add__(self, other):
        return BankCard(self.total_sum + other.total_sum, max(self.balance_limit, other.balance_limit))

    def __str__(self):
        return "To learn the balance call balance."

    @property
    def balance(self):
        if self.balance_limit is None:
            return int(self.total_sum)
        elif self.balance_limit > 0:
            self.balance_limit -= 1
            return int(self.total_sum)
        else:
            print("Balance check limits exceeded.")
            raise ValueError

    def put(self, sum_put):
        self.total_sum += sum_put
        print("You put sum_put dollars.")
