import streamlit as st
from streamlit_option_menu import option_menu
from streamlit.elements.image import image_to_url
import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar,Pie
from streamlit_echarts import st_pyecharts

def find_outliers_by_3segama(data,fea):
    data_std=np.std(data[fea])
    data_mean=np.mean(data[fea])
    outliers_cut_off=data_std*3
    lower_rule=data_mean-outliers_cut_off
    upper_rule=data_mean+outliers_cut_off
    data[fea+'_outliers']=data[fea].apply(lambda x:str('异常值')if x > upper_rule or x < lower_rule else '正常值')
    return data

def chart1(data):
    temp = df.groupby(data)[data].count()

    c = (Pie()
         .add("",  [list(z) for z in zip(temp.index.astype(str).tolist(),temp.values.tolist())])
         .set_global_opts(
        # title_opts=opts.TitleOpts(title="Pie-调整位置"),
    )
         .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
         )
    st_pyecharts(c)

st.set_page_config(page_title="金融风控模型展示", page_icon="💳", layout="wide")
img_url=image_to_url('/Users/penneyye/Desktop/Untitled/this.jpeg',width=-3,clamp=False,channels='RGB',output_format='auto',image_id='',allow_emoji=False)
st.markdown('''
<style>
.css-fg4pbf{background-image:url('''+img_url+''');}</style>
''',unsafe_allow_html=True)

sysmenu='''
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
'''
st.markdown(sysmenu,unsafe_allow_html=True)

with st.sidebar:
    file = st.sidebar.file_uploader("请上传文件", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.success("数据成功导入！")
        choose = option_menu("金融风控", ["数据概览", "数据可视化", "数据探索", "机器学习","结果展示"],
                             icons=["border", "bar-chart", "zoom-in", "cpu","archive"],
                             menu_icon="", default_index=0)

if choose == "数据概览":

    st.title("数据总体概括：")
    row0_0, row0_1=st.columns((1,3))
    with row0_1:

        st.subheader("描述性分析")
        st.write(df.describe())
        st.subheader("数据信息")
        data=pd.DataFrame({'特征': df.columns,
                          '数量': df.notnull().sum().values,
                          '类数': df.nunique().values,
                          '类型': df.dtypes.values})
        st.table(data.astype(str))
    with row0_0:
            st.subheader("数据尺寸")
            st.write("样本量：" + str(df.shape[0]))
            st.write("特征量：" + str(df.shape[1]))
            st.subheader("特征一览")
            st.write("id：为贷款清单分配的唯一信用证标识")
            "loanAmnt：贷款金额"
            "term：贷款期限（year）"
            "interestRate：贷款利率"
            "installment：分期付款金额"
            "grade：贷款等级"
            "subGrade：贷款等级之子级"
            "employmentTitle：就业职称"
            "employmentLength：就业年限（年）"
            "homeOwnership：借款人在登记时提供的房屋所有权状况"
            "annualIncome：年收入"
            "verificationStatus：验证状态"
            "issueDate：贷款发放的月份"
            "purpose：借款人在贷款申请时的贷款用途类别"
            "postCode：借款人在贷款申请中提供的邮政编码的前3位数字"
            "regionCode：地区编码"
            "dti：债务收入比"
            "delinquency_2years：借款人过去2年信用档案中逾期30天以上的违约事件数"
            "ficoRangeLow：借款人在贷款发放时的fico所属的下限范围"
            "ficoRangeHigh：借款人在贷款发放时的fico所属的上限范围"
            "openAcc：借款人信用档案中未结信用额度的数量"
            "pubRec：贬损公共记录的数量"
            "pubRecBankruptcies：公开记录清除的数量"
            "revolBal：信贷周转余额合计"
            "revolUtil：循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额"
            "totalAcc：借款人信用档案中当前的信用额度总数"
            "initialListStatus：贷款的初始列表状态"
            "applicationType：表明贷款是个人申请还是与两个共同借款人的联合申请"
            "earliesCreditLine：借款人最早报告的信用额度开立的月份"
            "title：借款人提供的贷款名称"
            "policyCode：公开可用的策略代码=1新产品不公开可用的策略代码=2"
            "n系列匿名特征：匿名特征n0-n14，为一些贷款人行为计数特征的处理"

elif choose == "数据可视化":

    st.title("数据可视化")

    row1_0, row1_spacer1, row1_1=st.columns((1,0.1,1))
    with row1_0:
        missing = df.isnull().sum() / len(df)
        missing = missing[missing > 0]
        missing = missing.sort_values(inplace=False)

        a=(Bar()
        .add_xaxis(missing.index.tolist())
        .add_yaxis("数据",missing.values.tolist())
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),
            title_opts=opts.TitleOpts(title="Bar-旋转X轴标签", subtitle="解决标签名字过长的问题"),
        ))
        st_pyecharts(a)



    with row1_1:

        numerical_fea = list(df.select_dtypes(exclude=['object']).columns)
        numerical_fea.remove('isDefault')
        for fea in numerical_fea:
            data_yi = find_outliers_by_3segama(df, fea)
        abnormal = pd.Series((data_yi[data_yi.columns[47:88]] == '异常值').sum().values, index=numerical_fea)
        abnormal = abnormal[abnormal > 0]
        abnormal = abnormal.sort_values(inplace=False)
        b=(Bar()
           .add_xaxis(abnormal.index.tolist())
            .add_yaxis("数据",abnormal.values.tolist())
               .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),
                title_opts=opts.TitleOpts(title="Bar-DataZoom（slider+inside）"),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
            )
               )
        st_pyecharts(b)

    st.header("离散型数据")
    row2_0, row2_spacer1, row2_1 = st.columns((1, 0.1, 2))
    with row2_0:
        label1=('grade','purpose','employmentLength','homeOwnership','verificationStatus',
                    'pubRecBankruptcies','n11','n12')

        select1=st.selectbox("选择你要探索的离散数据",label1,key="select1")

    with row2_1:
        st.subheader(select1+'饼图')
        chart1(select1)

    st.header("连续型数据")
    row3_0, row3_spacer1, row3_1 = st.columns((1, 0.1, 2))
    with row3_0:
        label2=('loanAmnt','interestRate','installment','employmentTitle',
                    'annualIncome','postCode','dti','revolBal','revolUtil',
                    'totalAcc','earliesCreditLine','title',)
        select2 = st.selectbox("选择你要探索的连续数据", label2, key="select1")


    with row3_1:
        d=(Bar()
            .add_xaxis(df['annualIncome'].value_counts().sort_index().index.tolist())
            .add_yaxis('数据',df['loanAmnt'].value_counts().sort_index().values.tolist(),
                          category_gap=0.1,color='b')

               )
        st_pyecharts(d)








elif choose == "数据探索":
    st.write("sifse")
elif choose == "机器学习":
    st.write("sief")
elif choose == "结果展示":
    st.write("sfs")