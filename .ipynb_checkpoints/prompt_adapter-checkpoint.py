import re
from typing import Tuple, Set, Optional, List, Dict, Union

#possibility
#result
def _get_condition(possibility:float, result:str) -> str:
    if possibility < 1.0:
        return str(possibility*100) + '% chance to be ' + result
    else:
        return str(possibility) + '% chance to be ' + result
'''
Category: Estrogen Status, category of [False, True], if it is False, why these instances have 97% possibility to be dead ?
Continuous: Non-industry income and expenditure/revenue, which is range from 0.0 to 1.0, if  this feature of X is < 0.3, why X have a 95% chance to be bankrupt?
for category example:
feature='Estrogen Status', val_range = tuple/list for continous/category, det='<','=','>', val='False', why X have a condition='90% possibility to be bankrupt' + '?'
(possibility, result) for condition
'''
def _get_query(feature, val_range, det, val, condition) -> str:
    query = ''
    possibility, result = condition['possibility'], condition['result']
    if type(val_range) is tuple:
        query = feature + ', range from ' + str(val_range[0]) + ' to ' + str(val_range[-1]) + ', if this feature of X is ' + str(det) + ' ' + str(val) + ', why X have a ' + str(_get_condition(possibility, result)) + '? '
    else:
        query = feature + ', category of ' + str(val_range) + ', if this feature of X is ' + str(val) + ', why X have a ' + str(_get_condition(possibility, result)) + '? '
    return query

"""
This function return the explanation for node splitting
Example:
desc='This is a Glass Identification Data Set from UCI. It contains 10 attributes with unit measurement expect RI that weight \
percent in corresponding oxide... '
role='doctor'
query={
    'feature':'Al Aluminum',
    'val_range':(0.1, 0.5),
    'det':'<',
    'val':0.25,
    'condition':{'possibility':97.3, 'result':'building_windows_non_float_processed'}
}
premise='Ba < 0.335, which indicates that the glass has a low barium content, suggesting it is less likely to be a type of glass \
that requires a high density or high refractive index, such as certain types of optical glass, and ...'
Returns:
    string: prompt text
"""
def get_explanation_prompt(desc:str, role:str, query:str, premise=None) -> str:
    prompt = ''
    query_ = _get_query(
        feature=query['feature'],
        val_range=query['val_range'],
        det=query['det'],
        val=query['val'],
        condition=query['condition']
    )
    if premise:
        prompt = 'Question: ' + desc + ' Given a group of instances X satisfy that:' + premise + ' Assume you are a ' + role + ', please explain that \n' + query_ + '\nYour explanation: #Please give a brief answer in one paragraph shorter\n'
    else:
        prompt = 'Question: ' + desc + ' Assume you are a ' + role + 'given a group of instances X, please explain that \n' + query_ + '\nYour explanation: #Please give a brief answer in one paragraph shorter \n'
    return prompt

#format like:
#{A}
#explanation
#input: 'in the context of ...'
#return: status, (tag, option(string))/None 
def get_explanation_result(response:str) -> Tuple[bool, Union[str, None]]:
    if len(response) > 0:
        return True, response  # Return the content inside the curly braces
    return False, None

#1. IRIS
#2. BREAST CANCER
#3. COMPANY BANKRUPTCY
#4. CHEMIST
#5. COMPUTER SCIENCE
explanation_shots = """
Question: The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Assume you are an botanist, given a group of instances X, please explain that petal width, range from 0.1 to 2.5 (cm), if this feature of X is < 0.8, why X are most likely to be setosa ?
Your explanation:
Iris Setosa is typically distinguished by its smaller petal width, usually less than 0.8 cm, compared to Iris Versicolor and Iris Virginica. This characteristic likely results from evolutionary adaptations to specific ecological niches or pollination strategies. As such, a petal width below 0.8 cm is a strong indicator of Iris Setosa, aiding botanists in species classification.


Question: This dataset of breast cancer patients was obtained from the 2017 November update of the SEER Program of the NCI, which provides information on population-based cancer statistics. The dataset involved female patients with infiltrating duct and lobular carcinoma breast cancer (SEER primary cites recode NOS histology codes 8522/3) diagnosed in 2006-2010. Patients with unknown tumour size, examined regional LNs, positive regional LNs, and patients whose survival months were less than 1 month were excluded; thus, 4024 patients were ultimately included. Given a group of instances X satisfy that: Survival Months (range from 1 to 107) is a statistical feature that means how long a patient has survived since being diagnosed. For early-stage breast cancer, where the cancer is localized and has not spread beyond the breast or nearby lymph nodes, the 5-year survival rate is relatively high. Therefore, instances of Survival Months <= 47.5 are likely to be benign. Assume you are an expert of Breast surgeons, please explain that Estrogen Status, category of [False, True], if this feature of X is False, why X have a 97% chance to be dead ?
Your explanation:
Estrogen Receptor-negative (ER-negative) tumors, indicated by "False" estrogen status, do not have receptors for estrogen and therefore do not respond to hormonal therapies like tamoxifen or aromatase inhibitors. These therapies are effective in treating estrogen receptor-positive (ER-positive) tumors, which can grow in response to estrogen and often have better survival outcomes due to the efficacy of these treatments. ER-negative breast cancers, however, are more challenging to treat because they require alternative therapeutic strategies such as chemotherapy, which may not be as effective. Consequently, ER-negative tumors tend to be more aggressive and are often associated with poorer prognoses and higher mortality rates. In the given dataset, where survival months range from 1 to 107 and instances with survival months ≤ 47.5 are considered likely to be less aggressive, the strong correlation between ER-negative status and a 97% chance of mortality underscores the aggressive nature and treatment challenges of ER-negative breast cancer, resulting in poorer survival outcomes.


Question: Company Bankruptcy Prediction dataset were built from the Taiwan Economic Jouranl for the years 1999 to 2009. Company bankruptcy was defined based on the business regulations of the Taiwan Stock Exchange. Given a group of instances X satisfy that: The low EPS-Net Income indicates weak profitability, while the low Quick Ratio suggests serious liquidity problems. Together, these factors could point to a higher risk of financial distress or bankruptcy. Assume you are a economist, please explain that Non-industry income and expenditure/revenue: Net Non-operating Income Ratio, which is range from 0.0 to 1.0, if this feature of X < 0.3, why X have a 95% chance to be bankrupt?
Your explanation:
A Net Non-operating Income Ratio of less than 0.3 indicates a high probability of bankruptcy because it suggests the company heavily depends on its core operations for revenue without significant contributions from non-operating activities. This lack of diversification means the company doesn't have a financial buffer to rely on during times of operational distress, which is exacerbated when combined with low EPS and a low Quick Ratio, indicating weak profitability and serious liquidity issues. Without substantial non-operating income to support cash flow and meet liabilities, the company is at a heightened risk of insolvency, especially when historical data shows that such financial profiles frequently lead to bankruptcy.


Question: This is a Glass Identification Data Set from UCI. It contains 10 attributes with unit measurement expect RI that weight percent in corresponding oxide. The response is glass type containing building_windows_float_processed, building_windows_non_float_processed, vehicle_windows_float_processed, containers, tableware and headlamps. Given a group of instances X satisfy that: Ba < 0.335, which indicates that the glass has a low barium content, suggesting it is less likely to be a type of glass that requires a high density or high refractive index, such as certain types of optical glass, and Mg > 2.78, which suggests that the glass may be designed for improved chemical durability and resistance to weathering. Magnesium is not typically present in high amounts in standard soda-lime glass, which is commonly used in windows. Those glass of above attributes are likely to be non-float processed. Assume you are a chemist, please explain that Al Aluminum, range from 0.29 to 3.5, if this feature of X is > 1.42, why X have a 70% chance to be building_windows_non_float_processed ?
Your explanation:
In Glass Identification, instances with low barium content (Ba < 0.335) and high magnesium content (Mg > 2.78) suggest glass that does not require high density or refractive index, and instead is designed for improved chemical durability and resistance to weathering. These characteristics are less typical of standard soda-lime glass used in windows, indicating a likelihood of being non-float processed. When the aluminum content (Al) is greater than 1.42, there's a 70% probability that the glass is used for building windows that are non-float processed. This is because higher aluminum content enhances chemical durability, thermal stability, and resistance to weathering—properties essential for building materials exposed to environmental elements. Non-float processed glass might require such compositions due to different manufacturing techniques that do not use molten metal beds, thereby necessitating a formulation with higher aluminum for achieving desired durability and stability.


Question: The target for this dataset is to build a network intrusion detector, a predictive model capable of distinguishing between bad connections, called intrusions or attacks, and normal connections. Attacks fall into four main categories: normal; DOS: denial-of-service; probe; R2L: unauthorized access from a remote machine, e.g. guessing password; Given a group of instances X satisfy that: num_compromised < 0.5, which is range from 0.0 to 1, and diff_srv_rate (% of connections to different services, is Traffic features computed using a two-second time window.), which is range from 0.0 to 1.0. This group of instances is likely to be probe attacks as it often involve scanning or reconnaissance activities aimed at gathering information about the network or services available. Assume you are a expert of computer science, please explain that src_bytes, number of data bytes from source to destination, range from 0 to 50000, if this feature of X is less than 7, why X are nearly certain be probe attack ?
Your explanation:
In network intrusion detection, probe attacks are characterized by scanning or reconnaissance activities aimed at gathering information about the network or available services. These activities typically involve sending minimal data to various parts of the network to identify weaknesses or open ports. The attribute `src_bytes`, representing the number of data bytes sent from source to destination, often provides insights into the nature of the traffic. Probe attacks usually result in low `src_bytes` values because the goal is to gather information rather than transfer large amounts of data. When `src_bytes` is less than 7, it suggests minimal data transfer, indicative of simple connection attempts or requests, such as basic pings or SYN packets, which are typical of probing activities. Additionally, the conditions of `num_compromised < 0.5` and low `diff_srv_rate` support this inference, as they indicate a lack of significant compromise signs and low diversity in service access, respectively. Instances with `src_bytes` less than 7 are highly indicative of probe attacks.

"""


'''
option={
    'feature':'Al Aluminum',
    'val_range':(0.1, 0.5),
    'det':'<',
    'val':0.25,
    'condition':{'possibility':97.3, 'result':'building_windows_non_float_processed'}
}
'''
def _get_option(feature, val_range, det, val, condition) -> str:
    opt = ''
    possibility, result = condition['possibility'], condition['result']
    if type(val_range) is tuple:
        opt = feature + ', range from ' + str(val_range[0]) + ' to ' + str(val_range[-1]) + ', if this feature of X is ' + str(det) + ' ' + str(val) + ', X have a ' + str(_get_condition(possibility, result)) + '.'
    else:
        opt = feature + ', category of ' + str(val_range) + ', if this feature of X is ' + str(val) + ', X have a ' + str(_get_condition(possibility, result)) + '.'
    return opt

'''
example:
opts = [
    'RI refractive index, range from 1.51 to 1.53, if this feature of X is < 1.517, X have a 73% chance to be windows_non_float_processed.',
    'Ca Calcium, range from 5.4 to 16.2, if this feature of X is > 8.235, X have a 50% chance to be windows_float_processed.',
    'Al Aluminum, range from 0.29 to 3.5, if this feature of X is > 1.42, X have a 70% chance to be windows_non_float_processed.',
    'K Potassium, range from 0.0 to 6.21, if this feature of X is > 0.03, X have a 70% chance to be tableware.'
]
return:
    dict:{key, desc}, text of options
'''
def options2text(opts:List, selection_keys:List) -> Tuple[Dict, str]:
    selection = {}
    selection_text = ''
    assert len(opts) <= len(selection_keys)
    for index,op in enumerate(opts):
        key = selection_keys[index]
        selection[key] = op
        if index < len(opts)-1:
            selection_text += key + '. ' + op + '\n'
        else:
            selection_text += key + '. ' + op
    return selection,selection_text

'''
This function return the explanation for node splitting
Example:
desc='This is a Glass Identification Data Set from UCI. It contains 10 attributes with unit measurement expect RI that weight \
percent in corresponding oxide. The response is glass type containing building_windows_float_processed, \
building_windows_non_float_processed, vehicle_windows_float_processed, containers, tableware and headlamps.'
role='doctor'
options: options text from function options2text
premise='Ba < 0.335, which indicates that the glass has a low barium content, suggesting it is less likely to be a type of glass \
that requires a high density or high refractive index, such as certain types of optical glass, and Mg > 2.78, which suggests \
that the glass may be designed for improved chemical durability and resistance to weathering. Magnesium is not typically present \
in high amounts in standard soda-lime glass, which is commonly used in windows. Those glass of above attributes are \
likely to be non-float processed.'
Returns:
    string: prompt text
'''
def get_selection_prompt(desc, role, options, premise=None) -> str:
    prompt = ''
    if premise:
        prompt = 'Question: ' + desc + ' Given a group of instances X satisfy that:' + premise + ' Assume you are a ' + role + ', which options is the best for further classification of X?\n' \
            + options + '\nYour choice:\n'
    else:
        prompt = 'Question: ' + desc + ' Assume you are a ' + role + 'given a group of instances X, which options is the best for further classification?\n' \
            + options + '\nYour choice:\n'
    return prompt

#format like:
#{A}
#explanation
#input: tag:'ABCDEF', response: '{A}\nfrom my knowledge...'
#return: status, (tag, option(string))/None 
def get_selection_result(selection_keys: Set[str], response:str) -> Tuple[bool, Union[Tuple[str, str], None]]:
    match = re.search(r'\{(.+?)\}', response)
    if match:
        selection = match.group(1)
        explanation = response[match.span()[-1]:].strip()
        if selection in selection_keys and len(explanation) > 0:
            return True, (selection, explanation)  # Return the content inside the curly braces
    return False, None

#1. IRIS
#2. BREAST CANCER
#3. COMPANY BANKRUPTCY
#4. CHEMIST
#5. COMPUTER SCIENCE
selection_shots = """ 
Question: The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Assume you are a botanist, given a group of instances X, which options is the best for further classification?
A. sepal length, range from 4.3 to 7.9 (cm), if this feature of X is < 5.45, X are most likely to be setosa.
B. petal width, range from 0.1 to 2.5 (cm), if this feature of X is < 0.8, X are most likely to be setosa.
C. petal length, range from 1.0 to 6.9 (cm), if this feature of X is < 2.35, X are most likely to be setosa.
Your choice:
{B}
Base on my knowledge, the average petal length of Iris setosa is approximately 1.4 to 4.9 centimeters, while the width ranges from 0.2 to 0.7 centimeters. In option b, a petal with a width less than 0.8cm is judged to be iris setosa, which conforms to the range of 0.2 to 0.7 width mentioned in the known information. Thus, option B seems to be the most likely correct statement.


Question: This dataset of breast cancer patients was obtained from the 2017 November update of the SEER Program of the NCI, which provides information on population-based cancer statistics. The dataset involved female patients with infiltrating duct and lobular carcinoma breast cancer (SEER primary cites recode NOS histology codes 8522/3) diagnosed in 2006-2010. Patients with unknown tumour size, examined regional LNs, positive regional LNs, and patients whose survival months were less than 1 month were excluded; thus, 4024 patients were ultimately included. Given a group of instances X satisfy that: Survival Months (range from 1 to 107) is a statistical feature that means how long a patient has survived since being diagnosed. For early-stage breast cancer, where the cancer is localized and has not spread beyond the breast or nearby lymph nodes, the 5-year survival rate is relatively high. Therefore, instances of Survival Months <= 47.5 are likely to be benign. Assume you are an expert of Breast surgeons, which options is the best for further classification of X?
A. N Stage, Adjusted AJCC 6th N, category of [N1, N2, N3], if this feature of X is N1, X have a 83% chance to be dead.
B. Age, range from 30 to 69, if this feature of X is > 61.5, X have a 94% chance to be dead.
C. Reginol Node Positive, range from 1 to 46, if this feature of X is > 2.5, X have a 80% chance to be dead.
D. Estrogen Status, category of [False, True], if this feature of X is False, X have a 97% chance to be dead.
Your choice:
{D} 
From the question we know that this group of patient are likely to be benign, doctors need to focus on the feature that which is most significative of malignant. N stage helps in determining the overall stage of cancer, along with the T (tumor) stage and M (metastasis) stage, which guide treatment decisions and prognosis for patients with cancer. Age can also be an important factor in the diagnosis and treatment of breast cancer. Regional node positivity suggests that the cancer has spread beyond the original site in the breast, which may impact the choice of therapies such as surgery, chemotherapy, radiation, or targeted treatments. Among them, Estrogen is the most important feature in diagnosing breast cancer because the majority of breast cancers are estrogen receptor-positive (ER+). When breast cancer cells have receptors for estrogen, they rely on this hormone to grow and divide. Therefore, understanding the role of estrogen in breast cancer is crucial for effective diagnosis and treatment planning. Thus, option D seems to be the most likely related to the classification.


Question: Company Bankruptcy Prediction dataset were built from the Taiwan Economic Jouranl for the years 1999 to 2009. Company bankruptcy was defined based on the business regulations of the Taiwan Stock Exchange. Given a group of instances X satisfy that: The low EPS-Net Income indicates weak profitability, while the low Quick Ratio suggests serious liquidity problems. Together, these factors could point to a higher risk of financial distress or bankruptcy. Assume you are a economist, which options is the best for further classification of X?
A. Non-industry income and expenditure/revenue: Net Non-operating Income Ratio, which is range from 0.0 to 1.0, if this feature of X is < 0.3, X have a 95% chance to be bankrupt.
B. Continuous interest rate (after tax): Net Income-Exclude Disposal Gain or Loss/Net Sales, which is range from 0.0 to 1.0, if this feature of X is > 0.78, X have a 0% chance to be bankrupt.
C. After-tax net Interest Rate: Net Income/Net Sales, which is range from 0.0 to 1.0, if this feature of X is < 0.81, X have a 63% chance to be bankrupt.
D. Pre-tax net Interest Rate: Pre-Tax Income/Net Sales, which is range from 0.0 to 1.0, if this feature of X is > 0.80, X have a 100% chance to be bankrupt.
Your choice:
{A}
As we know that this group of instances have a higher risk of bankruptcy, we should focus on the option which may related to these instances. The option A Non-industry income and expenditure/revenue ≤ 0.3 indicates that up to 30% of revenue comes from non-core activities. Option B Continuous interest rate (after tax) > 0.78 suggests strong profitability from core operations, excluding one-time items. This is generally positive for financial health and does not indicate bankruptcy risk. Option C and D refers to a net profit and profitability before taxes, which could indicate financial stress but not necessarily bankruptcy. Base on the given info, option A seems to be the most critical warning sign for potential bankruptcy comparing with other options. 


Question: This is a Glass Identification Data Set from UCI. It contains 10 attributes with unit measurement expect RI that weight percent in corresponding oxide. The response is glass type containing building_windows_float_processed, building_windows_non_float_processed, vehicle_windows_float_processed, containers, tableware and headlamps. Given a group of instances X satisfy that: Ba < 0.335, which indicates that the glass has a low barium content, suggesting it is less likely to be a type of glass that requires a high density or high refractive index, such as certain types of optical glass, and Mg > 2.78, which suggests that the glass may be designed for improved chemical durability and resistance to weathering. Magnesium is not typically present in high amounts in standard soda-lime glass, which is commonly used in windows. Those glass of above attributes are likely to be non-float processed. Assume you are a chemist, which options is the best for further classification of X?
A. RI refractive index, range from 1.51 to 1.53, if this feature of X is < 1.517, X have a 73% chance to be building_windows_non_float_processed.
B. Ca Calcium, range from 5.4 to 16.2, if this feature of X is > 8.235, X have a 50% chance to be building_windows_float_processed.
C. Al Aluminum, range from 0.29 to 3.5, if this feature of X is > 1.42, it have a 70% chance to be building_windows_non_float_processed.
D. K Potassium, range from 0.0 to 6.21, if this feature of X is < 0.03, it have a 70% chance to be tableware.
Your choice:
{C} 
The option A Refractive Index: Suggests "building_windows_non_float_processed" with 73% probability if RI < 1.517. However, this is less relevant since the glass is unlikely to be float processed. The option B Calcium: Indicates a 50% chance of "building_windows_float_processed" if Ca > 8.235, but float processing is unlikely given the conditions. The option C Aluminum: Suggests "building_windows_non_float_processed" with 70% probability if Al > 1.42. This aligns well with the instances memtioned with the question, making it a strong candidate. Option D Potassium: Indicates a 70% chance of being "tableware" if K < 0.03. This is a plausible option given the characteristics. Conclusion: Option C (Aluminum) is the best choice, given its high probability and alignment with non-float processed glass.


Question: The target for this dataset is to build a network intrusion detector, a predictive model capable of distinguishing between bad connections, called intrusions or attacks, and normal connections. Attacks fall into four main categories: normal; DOS: denial-of-service; probe; R2L: unauthorized access from a remote machine, e.g. guessing password; Given a group of instances X satisfy that: num_compromised < 0.5, which is range from 0.0 to 1, and diff_srv_rate (% of connections to different services, is Traffic features computed using a two-second time window.), which is range from 0.0 to 1.0. This group of instances is likely to be probe attacks as it often involve scanning or reconnaissance activities aimed at gathering information about the network or services available. Assume you are a expert of computer science, which options is the best for further classification of X?
A. protocol_type_is_tcp: category of [False, True], if this feature of X is True, X are likely to be probe attack.
B. number of connections to the same host in the past two seconds, range from 1 to 500, if this feature of X is < 8.5, X are tend to be normal. 
C. dst_host_rerror_rate: % of connections that have REJ errors, range from 0.0 to 1, if this feature of X is > 0.02, X may well be probe attack.
D. is_guest_login: category of [False, True], if it is True, it likely to be R2L.
E. src_bytes, number of data bytes from source to destination, range from 0 to 50000, if this feature of X is < 7, X are nearly certain be probe attack.
Your choice:
{E} 
Let's examine the options base on This group of instances is possible to be probe attacks: A, protocol_type_is_tcp, while many probe attacks might use TCP, this feature alone is not highly discriminatory for probe attacks versus other types of connections. B, number of connections to the same host in the past two seconds, suggests that a lower number of connections might indicate normal activity, but it doesn't specifically identify probe attacks. C, dst_host_rerror_rate, indicates that a higher rate of REJ errors could suggest probe activity, but it is not definitive without additional context. D, is_guest_login, is more indicative of R2L attacks rather than probe attacks. Finally, Option E, src_bytes, where a low number of data bytes from source to destination (less than 7) being nearly certain to be a probe attack, aligns well with the typical behavior of probe attacks, which often involve sending minimal data to gather information. Given the context and the characteristics of probe attacks, option E (src_bytes) is the most specific and relevant feature for further classification of the given instances as probe attacks.


"""

#unknown word set
UNK_WORD = set(['unknown','unk','unknow','unkown','none','unidentified','unclear','uncertain','not sure','not available','unavailable','unfortunately','unsure','however','answer','your answer'])
REJECT_WORD = set(['sorry','i am sorry','i\'m sorry','sorry','sorry,','sorry.','unable','can\''])

direct_answer_prompt = """
Question: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
Answer: Washington, D.C.

Question: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
Answer: Bharoto Bhagyo Bidhata

Question: Who was the artist nominated for an award for You Drive Me Crazy?
Answer: Jason Allen Alexander

Question: What person born in Siegen influenced the work of Vincent Van Gogh?
Answer: Peter Paul Rubens

Question: What is the country close to Russia where Mikheil Saakashvii holds a government position?
Answer: Georgia

Question: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
Answer: Heroin

Please answer the following question directly. You just need to output the answer.
Question: %s
Answer: 
"""

def get_direct_answer_prompt(question:str) -> str:
    assert question is not None
    return direct_answer_prompt%question

def get_direct_answer_result(input_string:str) -> Tuple[bool, str]:
    if input_string is not None:
        if input_string.lower().strip() in UNK_WORD:
            return False, None
        if set(input_string.lower().split(' ')).intersection(REJECT_WORD):
            return False, None
        return True, input_string
    else:
        return False, None

cot_answer_prompt = """
Question: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
Answer: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Question: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
Answer: {Bharoto Bhagyo Bidhata} First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Question: Who was the artist nominated for an award for You Drive Me Crazy?
Answer: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Question: What person born in Siegen influenced the work of Vincent Van Gogh?
Answer: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Question: What is the country close to Russia where Mikheil Saakashvii holds a government position?
Answer: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Question: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
Answer: {Heroin} First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}.

Please solve the following question step by step. Note that you answer should be enclosed in curly brackets like {Answer}.  
Question: %s
Your Answer:
"""

def get_cot_prompt(question:str) -> str:
    assert question is not None
    return cot_answer_prompt%(question)

def get_cot_answer_result(input_string:str) -> Tuple[bool, str]:
    match = re.search(r"\{([^}]*)\}", input_string)
    if match:
        answer = match.group(1)
        if answer.lower().strip() in UNK_WORD:
            return False,None
        if set(answer.lower().split(' ')).intersection(REJECT_WORD):
            return False,None
        return True,answer
    return False,None