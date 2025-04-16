from pymed import PubMed
from Bio import Entrez
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DataFetcher:
    """ A class to fetch data from PubMed using the pymed library.

        Attributes:
            pubmed (PubMed): An instance of the PubMed class from pymed.
            email (str): The email address of the user.
    """

    def __init__(self, tool: str, email: str):
        """ Initializes the DataFetcher with the given tool name and email.

            Parameters:
                tool (str): The name of the tool using the API.
                email (str): The email address of the user.
        """
        self.pubmed = PubMed(tool=tool, email=email)
        self.email = email

    def fetch_data(self, query: str, max_results: int = 100, keys: set = None):
        """ Fetches data from PubMed based on the given query.

            Parameters:
                query (str): The search query for PubMed.
                max_results (int): The maximum number of results to fetch.
                keys (set): A set of keys to filter the fetched data.

            Returns:
                list: A list of dictionaries containing the fetched data filtered by the keys provided.
        """
        # If no keys are provided, use the default set of keys
        if keys is None:
            keys = {"abstract", "keywords", "title", "doi", "pubmed_id"}

        articles = self.pubmed.query(query, max_results=max_results)
        articles_dict = [article.toDict() for article in articles]
        filtered_articles = [
            {key: article[key] for key in keys if key in article}
            for article in articles_dict
        ]

        # this fetches full article text from PMC if available
        # for article in filtered_articles:
        #     if "pubmed_id" in article:
        #         full_text = self._fetch_full_article_text(article["pubmed_id"])
        #         article["full_text"] = full_text

        return filtered_articles
    
    def _fetch_full_article_text(self, pubmed_id):
        """Fetches the full text of an article given its PubMed ID, if available in the PMC database.

        Parameters:
            pubmed_id (str): The PubMed ID of the article.

        Returns:
            str or None: The full text of the article, or None if not available.
        """
        Entrez.email = self.email

        try:
            handle = Entrez.elink(dbfrom="pubmed", id=pubmed_id, linkname="pubmed_pmc")
            record = Entrez.read(handle)
            handle.close()

            pmcid = record[0]['LinkSetDb'][0]['Link'][0]['Id']

            handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
            content = handle.read()
            handle.close()

            if content.strip().startswith("<?xml"):
                parser = "lxml-xml"
            else:
                parser = "html.parser"

            soup = BeautifulSoup(content, parser)

            body = soup.find("body")
            if body:
                return body.get_text(separator="\n", strip=True)
            else:
                return None

        except (IndexError, KeyError, Exception) as e:
            return None


if __name__ == "__main__":
    # Example usage
    tool = "DataWizardsProj"
    email = "some@email.com"

    data_fetch = DataFetcher(tool=tool, email=email)
    results = data_fetch.fetch_data("mononucleosis", max_results=10)
    for article in results:
        print(article)
        print("\n")
        

    # Example of using TF-IDF and cosine similarity for the abstracts
    summaries = [
        article["abstract"] if "abstract" in article else ""
        for article in results if article.get("abstract") is not None
    ]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(summaries)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Cosine Similarity Matrix:")
    print(cosine_sim)
