import scrapy


class NewsSpiderSpider(scrapy.Spider):
    name = "news_spider"
    allowed_domains = ["pubmed.ncbi.nlm.nih.gov"]
    start_urls = ["https://pubmed.ncbi.nlm.nih.gov"]

    def parse(self, response):
        for article in response.css("div.article"):
            title = article.css("h2::text").get()
            content = article.css("p::text").getall()
            url = response.urljoin(article.css("a::attr(href)").get())

            if title and content:  # Ensure non-empty content
                yield {
                    "title": title,
                    "content": content,
                    "url": url,
                }