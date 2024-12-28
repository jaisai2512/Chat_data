from API import api
import Rules
import json
import streamlit as st
from json_correction import format_correction
def summary_gen(df):
  rules = Rules.get_column_properties(df)

  system_prompt = '''As a seasoned data analyst,your responsibility is to go through a dictionary given to you , understand it and ouput the information in the template given to you while following the rules below:  
  1.Generate a semantic_type (a single word) for each field, based on its values (e.g., company, city, number, supplier, location, gender, longitude, latitude, URL, IP address, zip code, email, etc.)
  2.ALWAYS specify the description
  '''
  new_template= '''
  "dataset_name": "string",
  "dataset_description": "string",
  "fields_properties": [
    {
      "Name": "string",
      "Description": "string",
      "data_type":"string"
      "Semantic_type": "string",
      "type_of_measurement" : "string",
      "mean": "number",  // INCLUDE ONLY IF APPLICABLE (NOT NULL)
      "number_of_nulls": "integer",
      "sample_elements": ["element_1", "element_2", "..."]  // A list of sample values for the field
    },
    // More fields...
  ]
}
'''
  information ='''
Nominal: If the data represents categories or labels that have no inherent order, even if numeric (e.g., codes, IDs, or categorical values).
Ordinal: If the data represents categories with a meaningful order but without consistent intervals (e.g., ranks or satisfaction levels).
Discrete: If the data represents countable, distinct numeric values (e.g., whole numbers with specific meanings like area codes or counts).
Continuous: If the data represents measurable, continuous values with consistent intervals (e.g., heights, temperatures, or time).
Discrete(Categorical) : If the data is discrete and is categorical (ex:age only has limited value in column like 20,50 and 70)
When classifying type of measurement, prioritize the semantic type and description to understand the data's purpose. For example, numeric values representing categorical or countable data like area codes should be classified as discrete, not continuous
'''
  
  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"""
    Please annotate the dictionary below using the provided instructions:
    {rules}
    Please Consider the below Context and sample elements for choosing type_of_measurement:
    {information}
    Output template:
    {new_template}
    Return the updated JSON dictionary directly, without any explanation.
    """},
]
  o_summary = api(messages)
  count = 0
  '''while count <2:
    try:
      count = count +1
      summary = json.loads(o_summary)
      break
    except:
      o_summary = format_correction(o_summary,new_template)
  if(count == 2):
    st.write("reupload")
    exit()'''
  system_prompt ='''As an experienced data analyst, your task is to create a structured dataset annotation based on the template given . Follow these Rules:
      1. Fill in the dataset title and description accurately, ensuring clarity about the dataset's purpose and context.
      2. For each field in the dataset:
        i)Provide a description of the field's properties which can be used for data analysis and should based on the json.
        ii) Be more descriptive about the semantic type 
        '''
  temp = '''
      Dataset Title(Bold text): 
      [Provide the title of the dataset here]

      Dataset Description(Bold text): 
      [Provide a brief description of the dataset, including its purpose and any relevant context]

      Fields(Bold text):
          - Field 1(Bold text): 
                A summary on this field covering all the properties
          - Field 2(Bold text): 
                 A summary on this field covering all the properties
            [Continue for additional fields as necessary]
            ...
      '''
  sum = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"""
    Please Create a summary based on the json given below:
    {o_summary}
    Ouput template:
    {temp}
    """},
]
  s = api(sum)
  st.write(s)

  #st.write(rules)
  return rules,s 
  
