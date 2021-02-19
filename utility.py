def load_dict_file(filepath, separator=','):
    """
    Loads the content of a file as a dictionary
    :param filepath: path of file to be loaded. Should include folders and file type.
    :param separator: optional separator between values (default: ',')
    :return: dictionary containing the content of the file
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        dictionary = {}
        for line in file.readlines():
            kv = line.split(separator)
            value = kv[1].replace('\n', '')
            key = int(kv[0]) if kv[0].isnumeric() else kv[0]
            dictionary[key] = value
    return dictionary


def save_dict_file(filepath, content, separator=','):
    """
    Saves content of list as a vector in a file, similar to a Word2Vec document.
    :param separator: separator between values
    :param filepath: path of file to save.
    :param content: list of content to save.
    :return: None
    """
    print('Saving file "' + filepath + '".')
    with open(filepath, "w", encoding='utf-8') as file:
        if isinstance(content, dict):
            for k, v in content.items():
                file.write(str(k) + separator + str(v) + '\n')
        else:
            for i, c in enumerate(content):
                file.write(str(i) + separator + str(c) + '\n')
    print('"' + filepath + '" has been saved.')


def load_pair_file(filepath, separator=','):
    with open(filepath, 'r', encoding='utf-8') as file:
        listen = []
        for line in file.readlines():
            kv = line.split(separator)
            value = kv[1].replace('\n', '')
            if len(value.split(' ')) > 1:
                value = value.split(' ')
            listen.append((int(kv[0]), value))
    return listen


def save_vector_file_nonunique(filepath, content, separator=','):
    print('Saving file "' + filepath + '".')
    with open(filepath, "w", encoding='utf-8') as file:
        for i, c in content:
            file.write(str(i) + separator + str(c) + '\n')
    print('"' + filepath + '" has been saved.')