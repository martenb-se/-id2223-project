class HopsworksNoAPIKey(Exception):
    """
    Raised when the API key has not been correctly configured
    """
    pass


class AISStreamNoAPIKey(Exception):
    """
    Raised when the API key has not been correctly configured
    """
    pass


class HuggingfaceNoAccessToken(Exception):
    """
    Raised when the access token has not been correctly configured
    """
    pass


class HopsworksLoginError(Exception):
    """
    Raised when failing to log in to Hopsworks
    """
    pass


class HuggingfaceLoginError(Exception):
    """
    Raised when failing to log in to Huggingface
    """
    pass


class HopsworksGetFeatureStoreError(Exception):
    """
    Raised when failing to log in to Hopsworks
    """
    pass


class HopsworksGetFeatureGroupError(Exception):
    """
    Raised when failing to get a Feature Group from Hopsworks
    """
    pass


class HopsworksQueryReadError(Exception):
    """
    Raised when failing to make a 'read' query from Hopsworks
    """
    pass


class HopsworksFeatureGroupInsertError(Exception):
    """
    Raised when failing to insert an entry into a Feature Group on Hopsworks
    """
    pass


class HuggingfaceGetSpaceInfoError(Exception):
    """
    Raised when failing to get space info from Huggingface
    """
    pass


class TimeoutException(Exception):
    """
    Raised when a timeout occurs.
    """
    pass


class GitHubGetBackfillCSVError(Exception):
    """
    Raised when failing to get a backfill CSV from GitHub
    """
    pass
