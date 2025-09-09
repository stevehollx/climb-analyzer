# Downloads Shuttle Radar Topography Mission (SRTM GL1) Global 30m (SRTM30m) topo data for the world
# Create an earthdata.nasa.gov account here: https://urs.earthdata.nasa.gov/users/new

import requests 
import os

username = input('Username for earthdata.nasa.gov')
password = input('Password: ')

class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'
    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

   # Overrides from the library to keep headers when redirected to or from
   # the NASA auth host.

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and redirect_parsed.hostname != self.AUTH_HOST and original_parsed.hostname != self.AUTH_HOST:
                del(headers['Authorization'])
        return

def _get_file(url):

  filename = url[url.rfind('/')+1:]  

  try:
      response = session.get(url, stream=True)
      print(str(response.status_code) + '\n')

      response.raise_for_status()  

      with open(os.path.join('srtm30m',filename), 'wb') as fd:
          for chunk in response.iter_content(chunk_size=1024*1024):
              fd.write(chunk)

  except requests.exceptions.HTTPError as e:
      # handle any errors here
      print(e)


# create session with the user credentials that will be used to authenticate access to the data

session = SessionWithHeaderRedirection(username, password)

try:
  if not os.path.isdir('srtm30m'):
    os.mkdir('srtm30m')
except:
  print('Error: Could not create ./srtm30m directory.')


try:
    with open('srtm30m_urls.txt', 'r') as file:
        line_count = 0
        s = 0
        for line in file:
            line_count += 1
    with open('srtm30m_urls.txt', 'r') as file:            
        for url in file:
            url = url.strip()
            _get_file(url)
            print(url)  
            print(str(s) + ' of ' + str(line_count))
            s += 1
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

