def tester(givenstring="Too short"):
    print(givenstring)


while True:
    user_input = input("Write something (quit ends): ")

    if user_input == "quit":
        break

    if len(user_input) < 10:
        tester()
    else:
        tester(user_input)

