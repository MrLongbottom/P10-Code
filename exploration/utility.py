def find_id_from_value(dictionary: dict, value: str, printouts: bool = True):
    id = None
    for key, val in dictionary.items():
        if value == val:
            id = key

    if id is None:
        if printouts:
            print(f"Did not find the value '{value}' in the metadata")
        exit()
    else:
        if printouts:
            print(f"Found '{value}' with ID: {id}")
        return id


def get_category_ids_from_names(id2category: dict, names: list):
    category_ids = []
    for name in names:
        category_ids.append(find_id_from_value(id2category, name, printouts=False))
    return category_ids


def row_distribution_normalization(matrix):
    normalized_matrix = matrix
    for row in range(normalized_matrix.shape[0]):
        row_sum = sum(normalized_matrix[row])
        normalized_matrix[row] = [val / row_sum for val in normalized_matrix[row]]
    return normalized_matrix
