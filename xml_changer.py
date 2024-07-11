import os
import xml.etree.ElementTree as ET

def update_xml_name_tag(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            filepath = os.path.join(directory, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name = obj.find('name')
                if name is not None and name.text == 'number_plate':
                    name.text = 'licence'
            
            tree.write(filepath)
            print(f"Updated {filename}")

# Przykład użycia:
directory = 'TensorFlow/workspace/training_demo/images/images'
update_xml_name_tag(directory)
