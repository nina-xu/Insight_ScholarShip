# Insight_ScholarShip
Boosting your institution's international student enrollment

US universities need international students to bring in revenue and diverse perspectives. The avaiable strategies out there to increase international student enrollment don't show any evidence regarding whether their strategies actually work. In this project I combined data from multiple sources in order to identify actionable features that would help universities increase the percent of international students enrolling in their insitution.

I scraped and merged the following data sources:

National Center for Education Statistics: https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx

US News: https://www.usnews.com/best-colleges

International Student: https://www.internationalstudent.com/international-financial-aid/


I build the following linear regression to predict the logit transformation of the percent of international students.

logit(% enrollment) = -2.86 + 0.32*Int_graduation_rate + 0.27*diversity_index + 0.17*avg_intl_financial_aid + 0.04*pct_intl_financial_aid
