import xml.etree.ElementTree as ET
import re


def parse_xml(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    return root


def get_article_id(path: str):
    return path.split('/')[-1]


def get_body(root: ET.Element):
    return ' '.join([x.text for x in root.find('.//{*}block') if re.match(r"[^\s]+.*", x.text) != None])


def get_category(root: ET.Element):
    return [x.attrib['content'] for x in root.find('.//{*}head') if 'SAXo-Category' in x.attrib.values()][0]


def get_author(root: ET.Element):
    return [x.attrib['content'] for x in root.find('.//{*}head') if 'SAXo-Author' in x.attrib.values()][0]


def get_taxonomy(root: ET.Element):
    return [x.attrib['content'] for x in root.find('.//{*}head') if 'SAXo-Taxonomy' in x.attrib.values()][0]


def get_headline(root: ET.Element):
    return ''.join([x.text for x in root.find('.//{*}hedline')])


def create_article(path: str):
    root = parse_xml(path)
    return {'id': get_article_id(path), 'headline': get_headline(root), 'body': get_body(root), 'category': get_category(root), 'author': get_author(root), 'taxonomy': get_taxonomy(root)}


if __name__ == '__main__':
    article = create_article("/home/simba/Desktop/Finished Data 2018/2018-01-01/TabletXML/00_01_1-_sektion_man_s001_01_forside__0101_201801010000_1009001634.xml")
    print(article)
