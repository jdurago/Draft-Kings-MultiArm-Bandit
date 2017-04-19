from roto_grinders_mlb_lineup import scrape as rg_mlb_scrape

scrape_dict = {
    'mlb_rotogrinders': rg_mlb_scrape

}

def scrape(source):
    return scrape_dict[source]()