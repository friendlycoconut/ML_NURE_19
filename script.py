import random


def get_training_and_testing_lists(list):
    jumbled_list = random.sample(list, len(list))
    return jumbled_list[:int(0.7*len(list))], jumbled_list[int(0.7*len(list)):]


def changer(list):
    for i in range(len(list)):
        if list[i].find('setosa') != -1:
            list[i] = list[i].replace('setosa', '1')
        elif list[i].find('versicolor') != -1:
            list[i] = list[i].replace('versicolor', '2')
        elif list[i].find('virginica') != -1:
            list[i] = list[i].replace('virginica', '3')
    return list


if __name__ == '__main__':

    # read from file and add each line to list (rstrip because it is \n terminated)
    with open('iris.data') as f:
        origin_list = [line.rstrip() for line in f]

    # print(origin_list)  # Debug
    training, testing = get_training_and_testing_lists(origin_list)
    # print(training)
    # print(testing)

    changed_list = changer(origin_list)
    print(changed_list)

