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


a = BankCard(10, 2)
print(a.balance)
print(a.balance_limit)  # 1
a(5)  # You spent 5 dollars.
print(a.total_sum)  # 5
print(a)  # To learn the balance call balance.
print(a.balance)  # 5
try:
    a(6)  # Not enough money to spend 6 dollars.
except ValueError:
    pass
a(5)  # You spent 5 dollars.
try:
    a.balance  # Balance check limits exceeded.
except ValueError:
    pass
a.put(2)  # You put 2 dollars.
print(a.total_sum)  # 2
