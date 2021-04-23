from exploration.metadata_documents import find_id_from_value
import preprocess.preprocessing as pre


def get_category_ids_from_names(id2category: dict, names: list):
    category_ids = []
    for name in names:
        category_ids.append(find_id_from_value(id2category, name))
    return category_ids


if __name__ == '__main__':
    id2category = pre.prepro_file_load('id2category', folder_name='full')
    geographic_category_names = ["Frederikshavn-avis", "Morsø Debat", "Morsø-avis", "Rebild-avis", "Brønderslev-avis",
                                 "Thisted-avis", "Jammerbugt-avis", "Vesthimmerland-avis", "Hjørring-avis",
                                 "Aalborg-avis", "Morsø Sport", "Thisted sport", "Mariagerfjord-avis", "Udland-avis"]
    geographic_category_ids = get_category_ids_from_names(id2category, geographic_category_names)
    topical_category_ids = [id for id in id2category.keys() if id not in geographic_category_ids]
