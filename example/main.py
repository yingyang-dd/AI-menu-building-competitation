from model import *
from tabulate import tabulate
import csv 

# menu url
menu_url = ['https://merchant-portal.doordash.com/onboarding/api/v1/platform/menuLink/Restaurant/3ab8d71a-9374-4323-a20c-f23da58fcfc6']

# run the model 
prediction = menu_builder_model.menu_builder(menu_url)
output_json = json.loads(prediction)

# store model outputs
is_valid_menu = output_json['is_valid_menu']
input_quality_score = str(output_json['input_quality'])
menu_complexity = output_json['menu_complexity']
confidence_score = str(output_json['confidence'])
menu_output = output_json['menu_output']

# post-processing    
menu_output = remove_empty_categories(remove_items_with_zero_or_null_price(menu_output))
menu_output_table = json_to_flat_format(menu_output)
menu_output_table.to_csv('example_output.csv', index=False)


print("menu url: {}".format(menu_url))
print("is_valid_menu: {}".format(is_valid_menu))
print("menu_complexity: {}".format(menu_complexity))
print("menu_output: ")
print(tabulate(menu_output_table, headers='keys', tablefmt='psql'))