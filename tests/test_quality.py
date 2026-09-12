"""内容质量（垃圾内容）过滤测试。

用例取自线上真实结果：
「科技新闻简报：m88JDB电子引领体育数字化革新」、
「正规买球万博APP科技新闻摘要合集」。
"""

from __future__ import annotations

from news_agent.quality import filter_spam, is_spam, spam_score


def test_brand_term_is_spam(article_factory):
    article = article_factory(
        "科技新闻简报：m88JDB电子引领体育数字化革新- 体坛网",
        summary="文章简报报道了m88JDB电子引领体育数字化革新的相关科技新闻。",
    )
    assert spam_score(article) == 1.0
    assert is_spam(article)


def test_multiple_spam_terms_accumulate_to_spam(article_factory):
    article = article_factory(
        "正规买球平台推荐：赔率与盘口分析",
        summary="提供下注与投注参考。",
    )
    assert is_spam(article)


def test_single_spam_term_is_tolerated(article_factory):
    """正当报道可能提到博彩（如监管、营收），单词命中不应误杀。"""
    article = article_factory(
        "澳门博彩业二季度营收同比增长",
        summary="监管机构表示将加强对赌场中介业务的合规审查。",
    )
    assert spam_score(article) < 0.6
    assert not is_spam(article)


def test_promo_plus_spam_terms_is_spam(article_factory):
    article = article_factory(
        "娱乐城APP下载：注册送彩金",
        summary="立即下载参与活动。",
    )
    assert is_spam(article)


def test_blacklisted_host_is_spam(article_factory):
    article = article_factory(
        "体育资讯速递",
        summary="今日赛事汇总。",
        url="https://www.1xbet-promo.example.com/news/1",
    )
    assert spam_score(article) == 1.0


def test_regular_news_is_not_spam(article_factory):
    for title, summary in [
        ("人形机器人量产在即，产业链持续升温", "多家公司公布量产计划。"),
        ("AI赋能历史经典产业 给千年技艺插上科技翅膀", "新华网报道。"),
        ("科技新闻", "风闻平台发布了关于2030年科技新闻的讨论。"),
    ]:
        article = article_factory(title, summary=summary)
        assert spam_score(article) == 0.0, title


def test_filter_spam_splits_input(article_factory):
    good = article_factory("人形机器人产业观察")
    bad = article_factory("正规买球万博APP科技新闻摘要合集")
    kept, dropped = filter_spam([good, bad])
    assert kept == [good]
    assert dropped == [bad]
