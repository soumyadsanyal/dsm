import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')
matplotlib.pylab.rcParams['figure.figsize'] = (10, 6)
import itertools
import scipy.stats as stats
import statsmodels.graphics.gofplots as gofplots
from pandas.tools.plotting import scatter_matrix
import statsmodels.stats.weightstats as weightstats

# definitions

data=pd.read_pickle("/home/soumya/research/insight/insight_project/modeling/data/dsm_data_scoped_variables.pkl")

interpretation={}

interpretation["categorical"]={
                1: "Yes",
                2: "No",
                -1: "Inapplicable",
                -2: "Determined in previous round",
                -7: "Refused",
                -8: "Don't know",
                -9: "Not ascertained",
                -10: "HOURLY WAGE >= $76.96",
                -13: "INITIAL WAGE IMPUTED"
                }

interpretation["RACE/ETHNICITY_(EDITED/IMPUTED)"]={1: "HISPANIC",
                        2: "NON-HISPANIC WHITE ONLY",
                        3: "NON-HISPANIC BLACK ONLY",
                        4: "NON-HISPANIC ASIAN ONLY",
                        5: "NON-HISPANIC OTHER RACE OR MULTIPLE RACE"
                        }

interpretation["SEX"]={1: "MALE", 2: "FEMALE"}

interpretation["MARITAL_STATUS-12/31/13_(EDITED/IMPUTED)"]={-9: "NOT ASCERTAINED",
                           -8: "DK",
                           -7: "REFUSED",
                           1: "MARRIED",
                           2: "WIDOWED",
                           3: "DIVORCED",
                           4: "SEPARATED",
                           5: "NEVER MARRIED",
                           6: "UNDER 16 - INAPPLICABLE"
                          }

interpretation["EDUCATION_RECODE_(EDITED)"]={-9: "NOT ASCERTAINED",
                             -8: "DK",
                             -7: "REFUSED",
                             -1: "INAPPLICABLE OR UNDER 5",
                             1: "LESS THAN/EQUAL TO 8TH GRADE",
                             2: "9 - 12TH GRADE, NO HS DIPLOMA OR GED",
                             13: "GED OR HS GRAD",
                             14: "BEYOND HS,COLLEGE(NO 4YR DEG),ASSOC DEG",
                             15: "4-YEAR COLLEGE DEGREE, BACHELOR'S DEGREE",
                             16: "MASTER'S, DOCTORATE, OR PROFESSIONAL DEG"
                             }

interpretation["INDUSTRY_GROUP_RD_3/1_CMJ"]={-9: "NOT ASCERTAINED",
                            -1: "INAPPLICABLE",
                            1: "NATURAL RESOURCES",
                            2: "MINING",
                            3: "CONSTRUCTION",
                            4: "MANUFACTURING",
                            5: "WHOLESALE AND RETAIL TRADE",
                            6: "TRANSPORTATION AND UTILITIES",
                            7: "INFORMATION",
                            8: "FINANCIAL ACTIVITIES",
                            9: "PROFESSIONAL AND BUSINESS SERVICES",
                            10: "EDUCATION, HEALTH, AND SOCIAL SERVICES",
                            11: "LEISURE AND HOSPITALITY",
                            12: "OTHER SERVICES",
                            13: "PUBLIC ADMINISTRATION",
                            14: "MILITARY",
                            15: "UNCLASSIFIABLE INDUSTRY"}

interpretation["OCCUPATION_GROUP_RD_3/1_CMJ"]={
    -9: "NOT ASCERTAINED",
    -1: "INAPPLICABLE",
    1: "MANAGEMENT, BUSINESS, AND FINANCIAL OPER",
    2: "PROFESSIONAL AND RELATED OCCUPATIONS", 
    3: "SERVICE OCCUPATIONS",
    4: "SALES AND RELATED OCCUPATIONS", 
    5: "OFFICE AND ADMINISTRATIVE SUPPORT",
    6: "FARMING, FISHING, AND FORESTRY",
    7: "CONSTRUCTION, EXTRACTION, AND MAINTENANC",
    8: "PRODUCTION, TRNSPORTATION, MATRL MOVING", 
    9: "MILITARY SPECIFIC OCCUPATIONS",
    11: "UNCLASSIFIABLE OCCUPATION"}

interpretation["CENSUS_REGION_AS_OF_12/31/13"]={-1: "Inapplicable",
                                                1:"Northeast",
                                                2:"Midwest",
                                                3:"South",
                                                4:"West"}

interpretation["EMPLOYMENT_STATUS_RD_3/1"]={-9: "NOT ASCERTAINED",
                                            -8: "DK",
                                            -7: "REFUSED",
                                            -1: "INAPPLICABLE",
                                            1: "EMPLOYED AT RD 3/1 INT DATE",
                                            2: "JOB TO RETURN TO AT RD 3/1 INT DATE",
                                            3: "JOB DURING RD 3/1 REF PERIOD",
                                            4: "NOT EMPLOYED DURING RD 3/1"}

interpretation["FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013"]={
     1: "<65 ANY PRIVATE",
     2: "<65 PUBLIC ONLY",
     3: "<65 UNINSURED",
     4: "65+ EDITED MEDICARE ONLY",
     5: "65+ EDITED MEDICARE AND PRIVATE",
     6: "65+ EDITED MEDICARE AND OTH PUB ONLY",
     7: "65+ UNINSURED",
     8: "65+ NO MEDICARE AND ANY PUBLIC/PRIVATE"
    }


 
variables=[]
with open("/home/soumya/research/insight/insight_project/modeling/code/dsm_exog_.txt","r") as f:
    lines=f.readlines()
    for line in lines:
        variables.append(line.strip().upper())

def check_discrete(data,thevariable):
    if len(data[thevariable].unique())>19 or "AGE_OF" in thevariable or "AGE_AS" in thevariable or "#" in thevariable:
        return "continuous"
    else:
        return "categorical"


continuous=[term for term in variables if check_discrete(data,term)=="continuous"]
categorical=[term for term in variables if check_discrete(data,term)=="categorical"]
bucketized=[term for term in variables if "BUCKET" in term]




health=[term for term in categorical if "DIAG" in term or 
        "LIMITATION" in term or
        "ASTHMA" in term or 
        "JOINT" in term or 
        "PEAK_FLOW" in term or
       "PREGNANT" in term or
       "ACUTE" in term or
       "BRONCH" in term]

demog=[term for term in categorical if "SEX" in term
		or "MARITAL" in term
		or "RACE" in term]


age=[term for term in continuous if "AGE_AS" in term]
age_diagnosed=[term for term in continuous if "AGE_OF" in term]
bmi=[term for term in continuous if "ADULT" in term and "BODY" in term and "MASS" in term]
incomes=[term for term in continuous if "FAMILY" in term and "INCOME" in term]
durations=['#_WKS/MON_WOUT_HLTH_INS_PRV_YR-PN_18_ONL']
utilizations=[term for term in continuous if "#" in term and "WKS" not in term]
expenditures=['TOTAL_OFFICE-BASED_EXP_13',
              'TOTAL_OUTPATIENT_PROVIDER_EXP_13',
              'TOT_HOSP_IP_FACILITY_+_DR_EXP_13',
              'TOTAL_ER_FACILITY_+_DR_EXP_13'
             ]

   
cancer=[term for term in health if "CANCER" in term]
lung=[term for term in health if "LUNG" in term or "BRONCH" in term or "ASTHMA" in term
     or "EMPHYSEMA" in term
     or "INH" in term]
cardio=[term for term in health if "BLOOD" in term or "CHOLESTEROL" in term or "CORON" in term or "HEART" in term
       or "STROKE" in term
       or "DIABETES" in term
       or "ANGINA" in term]
joint=[term for term in health if "JOINT" in term or "ARTHR" in term]
other=[term for term in health if term not in joint+cardio+lung+cancer]    

healthplus=["%s_PLUS"%term for term in health]
for term in health:
    data["%s_PLUS"%term]=data[term].map(lambda x: int(x==1))


employment_plus=['EMPLOYMENT_STATUS_RD_3/1_PLUS',
 'HAS_MORE_THAN_ONE_JOB_RD_3/1_INT_DATE_PLUS',
 'SELF-EMPLOYED_AT_RD_3/1_CMJ_PLUS',
 'CHOICE_OF_HEALTH_PLANS_AT_RD_3/1_CMJ_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_PLUS',
 'UNION_STATUS_AT_RD_3/1_CMJ_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_PLUS',
 'HEALTH_INSUR_HELD_FROM_RD_3/1_CMJ_(ED)_PLUS',
 'HEALTH_INSUR_OFFERED_BY_RD_3/1_CMJ_(ED)_PLUS',
 'EMPLOYER_OFFERS_HEALTH_INS_RD_3/1_CMJ_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_PLUS',
 'PRIVATE_INSURANCE_ANY_TIME_IN_R5/R3_PLUS',
 'PUBLIC_INS_ANY_TIME_IN_R5/R3_PLUS',
 'INSURED_ANY_TIME_IN_R3/1_PLUS',
 'ANY_TIME_COVERAGE_BY_STATE_INS_-_R3/1_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_NATURAL RESOURCES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_MINING_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_CONSTRUCTION_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_MANUFACTURING_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_WHOLESALE AND RETAIL TRADE_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_TRANSPORTATION AND UTILITIES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_INFORMATION_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_FINANCIAL ACTIVITIES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_PROFESSIONAL AND BUSINESS SERVICES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_EDUCATION, HEALTH, AND SOCIAL SERVICES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_LEISURE AND HOSPITALITY_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_OTHER SERVICES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_PUBLIC ADMINISTRATION_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_MILITARY_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_UNCLASSIFIABLE INDUSTRY_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_NOT ASCERTAINED_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_INAPPLICABLE_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_MANAGEMENT, BUSINESS, AND FINANCIAL OPER_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_PROFESSIONAL AND RELATED OCCUPATIONS_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_SERVICE OCCUPATIONS_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_SALES AND RELATED OCCUPATIONS_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_OFFICE AND ADMINISTRATIVE SUPPORT_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_FARMING, FISHING, AND FORESTRY_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_CONSTRUCTION, EXTRACTION, AND MAINTENANC_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_PRODUCTION, TRNSPORTATION, MATRL MOVING_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_MILITARY SPECIFIC OCCUPATIONS_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_UNCLASSIFIABLE OCCUPATION_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_NOT ASCERTAINED_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_INAPPLICABLE_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_<65 ANY PRIVATE_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_<65 PUBLIC ONLY_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_<65 UNINSURED_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_65+ EDITED MEDICARE ONLY_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_65+ EDITED MEDICARE AND PRIVATE_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_65+ EDITED MEDICARE AND OTH PUB ONLY_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_65+ UNINSURED_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_65+ NO MEDICARE AND ANY PUBLIC/PRIVATE_PLUS']

employment=[term for term in categorical if "GROUP" in term or
            "UNION" in term or 
            "INSUR" in term or 
            "EMPLOYER" in term or
            "INSURANCE" in term or
            "PUBLIC_INS" in term or
            "COVERAGE" in term or
            "CHOICE" in term or
            "EMPLOYED" in term or
            "EMPLOYMENT" in term or
            "JOB" in term or
            False
 ]

for term in employment:
    if "GROUP" in term or "INSURANCE_COVERAGE" in term:
        for key in interpretation[term]:
            #print(key, interpretation[term][key])
            data["%s_%s_PLUS"%(term,interpretation[term][key])]=data[term].map(lambda x: int(x==key))


data["EDUCATION_PLUS"]=data["EDUCATION_RECODE_(EDITED)"].map(lambda x: 0  if x<1 else x)

for term in health:
    data["%s_PLUS"%term]=data[term].map(lambda x: int(x==1))
    
healthplus=["%s_PLUS"%term for term in health]

data["SEX_MALE"]=data["SEX"].map(lambda x: int(x==1))
data["RACE_WHITE"]=data["RACE/ETHNICITY_(EDITED/IMPUTED)"].map(lambda x: int(x==2))
data["RACE_HISPANIC"]=data["RACE/ETHNICITY_(EDITED/IMPUTED)"].map(lambda x: int(x==1))
data["RACE_BLACK"]=data["RACE/ETHNICITY_(EDITED/IMPUTED)"].map(lambda x: int(x==3))
data["RACE_ASIAN"]=data["RACE/ETHNICITY_(EDITED/IMPUTED)"].map(lambda x: int(x==4))
data["RACE_OTHER_RACE"]=data["RACE/ETHNICITY_(EDITED/IMPUTED)"].map(lambda x: int(x==5))
data["MARITAL_MARRIED"]=data["MARITAL_STATUS-12/31/13_(EDITED/IMPUTED)"].map(lambda x: int(x==1))
data["MARITAL_WIDOWED"]=data["MARITAL_STATUS-12/31/13_(EDITED/IMPUTED)"].map(lambda x: int(x==2))
data["MARITAL_DIVORCED"]=data["MARITAL_STATUS-12/31/13_(EDITED/IMPUTED)"].map(lambda x: int(x==3))
data["MARITAL_SEPARATED"]=data["MARITAL_STATUS-12/31/13_(EDITED/IMPUTED)"].map(lambda x: int(x==4))
data["MARITAL_NEVER_MARRIED"]=data["MARITAL_STATUS-12/31/13_(EDITED/IMPUTED)"].map(lambda x: int(x==5))
data["MARITAL_MARRIAGE_INELIGIBLE"]=data["MARITAL_STATUS-12/31/13_(EDITED/IMPUTED)"].map(lambda x: int(x==6))

data["EDUCATION_PLUS"]=data["EDUCATION_RECODE_(EDITED)"].map(lambda x: 0  if x<1 else x)

demogplus=['SEX_MALE',
           'RACE_WHITE',
           'RACE_HISPANIC',
           'RACE_BLACK',
           'RACE_ASIAN',
           'RACE_OTHER_RACE',
           'MARITAL_MARRIED',
           'MARITAL_WIDOWED',
           'MARITAL_DIVORCED',
           'MARITAL_SEPARATED',
           'MARITAL_NEVER_MARRIED',
           'MARITAL_MARRIAGE_INELIGIBLE',
           'EDUCATION_PLUS']


for term in employment:
    if "GROUP" not in term or "INSURANCE_COVERAGE" not in term:
        data["%s_PLUS"%term]=data[term].map(lambda x: int(x==1))
        

employmentplus=['EMPLOYMENT_STATUS_RD_3/1_PLUS',
 'HAS_MORE_THAN_ONE_JOB_RD_3/1_INT_DATE_PLUS',
 'SELF-EMPLOYED_AT_RD_3/1_CMJ_PLUS',
 'CHOICE_OF_HEALTH_PLANS_AT_RD_3/1_CMJ_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_PLUS',
 'UNION_STATUS_AT_RD_3/1_CMJ_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_PLUS',
 'HEALTH_INSUR_HELD_FROM_RD_3/1_CMJ_(ED)_PLUS',
 'HEALTH_INSUR_OFFERED_BY_RD_3/1_CMJ_(ED)_PLUS',
 'EMPLOYER_OFFERS_HEALTH_INS_RD_3/1_CMJ_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_PLUS',
 'PRIVATE_INSURANCE_ANY_TIME_IN_R5/R3_PLUS',
 'PUBLIC_INS_ANY_TIME_IN_R5/R3_PLUS',
 'INSURED_ANY_TIME_IN_R3/1_PLUS',
 'ANY_TIME_COVERAGE_BY_STATE_INS_-_R3/1_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_NATURAL RESOURCES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_MINING_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_CONSTRUCTION_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_MANUFACTURING_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_WHOLESALE AND RETAIL TRADE_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_TRANSPORTATION AND UTILITIES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_INFORMATION_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_FINANCIAL ACTIVITIES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_PROFESSIONAL AND BUSINESS SERVICES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_EDUCATION, HEALTH, AND SOCIAL SERVICES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_LEISURE AND HOSPITALITY_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_OTHER SERVICES_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_PUBLIC ADMINISTRATION_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_MILITARY_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_UNCLASSIFIABLE INDUSTRY_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_NOT ASCERTAINED_PLUS',
 'INDUSTRY_GROUP_RD_3/1_CMJ_INAPPLICABLE_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_MANAGEMENT, BUSINESS, AND FINANCIAL OPER_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_PROFESSIONAL AND RELATED OCCUPATIONS_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_SERVICE OCCUPATIONS_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_SALES AND RELATED OCCUPATIONS_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_OFFICE AND ADMINISTRATIVE SUPPORT_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_FARMING, FISHING, AND FORESTRY_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_CONSTRUCTION, EXTRACTION, AND MAINTENANC_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_PRODUCTION, TRNSPORTATION, MATRL MOVING_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_MILITARY SPECIFIC OCCUPATIONS_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_UNCLASSIFIABLE OCCUPATION_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_NOT ASCERTAINED_PLUS',
 'OCCUPATION_GROUP_RD_3/1_CMJ_INAPPLICABLE_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_<65 ANY PRIVATE_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_<65 PUBLIC ONLY_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_<65 UNINSURED_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_65+ EDITED MEDICARE ONLY_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_65+ EDITED MEDICARE AND PRIVATE_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_65+ EDITED MEDICARE AND OTH PUB ONLY_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_65+ UNINSURED_PLUS',
 'FULL_YEAR_INSURANCE_COVERAGE_STATUS_2013_65+ NO MEDICARE AND ANY PUBLIC/PRIVATE_PLUS']






# functions

def interpret_vectorized(data,feature):
    if feature in interpretation:
        return data[feature].map(lambda x: interpretation[feature][x])
    elif feature in categorical:
        return data[feature].map(lambda x: interpretation["categorical"][x])
    else:
        return data[feature].map(lambda x: "inapplicable" if x<0 else x)


    temp=interpret_vectorized(data,thevar)
    w=data["FINAL_PERSON_WEIGHT_2013"]
    temp=pd.concat([temp,w],axis=1)
    temp["ones"]=1
    temp["scaled"]=temp["ones"]*temp["FINAL_PERSON_WEIGHT_2013"]
    result=temp.groupby([thevar])["scaled"].sum()
    ax=result.plot(kind="bar",title=thevar)
    ax.set_ylabel("Count")
    plt.show()

   
def zoom_hist(thevar, therange,bin_width=100, xticks=None):
    '''wrapper around df.hist'''
    temp=data[[thevar]+["FINAL_PERSON_WEIGHT_2013"]]
    bins=range(therange[0],therange[1],bin_width)
    ax=temp[thevar].hist(bins=bins,range=therange, weights=temp["FINAL_PERSON_WEIGHT_2013"])
    ax.set_title("%s between %s and %s, bin width is %s"%(thevar,therange[0],therange[1],bin_width))
    ax.set_xlabel(thevar)
    ax.set_ylabel("Count")
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation="vertical")
    plt.show()

def look_at_pairs(x, y, x_range, y_range, w, colorbar=False, gridsize=200, bins=100, mincount=None):
    x_min, x_max= x_range[0], x_range[1]
    y_min, y_max = y_range[0], y_range[1]
    plt.hexbin(x, y, C = w, gridsize = gridsize, bins = bins,
       xscale = 'linear', yscale = 'linear',
       cmap=None, norm=None, vmin=None, vmax=None,
       alpha=1, linewidths=None, edgecolors='none',
       reduce_C_function = np.sum, mincnt=mincount, marginals=False,
          )
    plt.ylim([y_min,y_max])
    plt.xlim([x_min,x_max])
    if colorbar:
        plt.colorbar()
    plt.show()


def four_correlations(x,
                      x_range,
                      w,
                      xticks=None,
                      office_range=[0,100000],
                      outpatient_range=[0,26000],
                      inpatient_range=[0,320000],
                      er_range=[0,70000],
                      colorbar=False, 
                      gridsize=200, 
                      bins=100, 
                      mincount=None):
    
    term=x.name
    plt.figure(figsize=(15,15))
    
    for pos,y in [(1,data["TOTAL_OFFICE-BASED_EXP_13"]),
                  (2,data["TOTAL_OUTPATIENT_PROVIDER_EXP_13"]),
                  (3, data["TOT_HOSP_IP_FACILITY_+_DR_EXP_13"]),
                  (4, data["TOTAL_ER_FACILITY_+_DR_EXP_13"]),
                 ]:
        plt.subplot(2,2,pos)
        plt.hexbin(x,
                   y,
                   C = w,
                   gridsize = gridsize,
                   bins = bins,
                   xscale = 'linear',
                   yscale = 'linear',
                   cmap=None,
                   norm=None,
                   vmin=None,
                   vmax=None,
                   alpha=1,
                   linewidths=None,
                   edgecolors='none',
                   reduce_C_function = np.sum,
                   mincnt=mincount,
                   marginals=False,
                   )
        plt.title(y.name)
        plt.ylim(office_range)
        plt.xlim(x_range)
        plt.xlabel(term)
        if xticks:
            plt.xticks(xticks[0],[thing.replace("NON-HISPANIC ","")[:15] for thing in xticks[1]], rotation=40)
        if colorbar:
            plt.colorbar()
        
        
    plt.tight_layout()
    plt.show()
    



