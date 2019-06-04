#!/usr/bin/env python
# coding: utf-8

# Author: Hongyi Chen


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import time
import json
import pandas as pd


# # Path where you save the webdriver 
# executable_path = '/Users/charleschen/Downloads/geckodriver'

# # initiator the webdriver for Firefox browser
# driver = webdriver.Firefox(executable_path=executable_path)

# driver.implicitly_wait(10)

# # send a request
# location = "Manhattan"
# driver.get("https://www.yelp.com/search?find_desc=Restaurants&find_loc="+ location+"&sortby=review_count")

# time.sleep(5)


# you should see a Firefox window open


# In[51]:


# Get all cafe links for a defined number of pages.
def getRestaurant(num_of_page):
        
    executable_path = '/Users/charleschen/Downloads/geckodriver'
    driver = webdriver.Firefox(executable_path=executable_path)
    driver.implicitly_wait(10)
    driver.get("https://www.yelp.com/search?find_desc=Restaurants&find_loc="+ location+"&sortby=review_count")
    time.sleep(5)
    
    # define a variable of list to store all cafe links.
    list_cafe_link = []

    #num_of_page = 5
    
    # loop for each page
    for page in range(num_of_page):             
        restaurant_name_list = []   
        above_100_review_cafe = []  
        cafe_name_list = []
        #cafe_name = []
        
        cafe_name = driver.find_elements_by_xpath("//a[@class = 'lemon--a__373c0__IEZFH link__373c0__29943 link-color--blue-dark__373c0__1mhJo link-size--inherit__373c0__2JXk5']")
        
        cafe_num_review = driver.find_elements_by_xpath("//span[@class = 'lemon--span__373c0__3997G text__373c0__2pB8f reviewCount__373c0__2r4xT text-color--mid__373c0__3G312 text-align--left__373c0__2pnx_']")
   
        #Get cafe names on one page
        for selenium in cafe_name[6:65]:
            #print(selenium.text)
            if not selenium.text =="read more":
                cafe_name_list.append(selenium.text)
                
        # filter out cafe with less than 100 reviews and below 200.
        for index in range(len(cafe_num_review[3:33])):
            if int(cafe_num_review[3:33][index].text.split()[0]) >= 000:
                above_100_review_cafe.append(cafe_name_list[index])
                
        # Get Cafe links from each name    
        try:          
            for cafe in above_100_review_cafe:
                cafe_link = driver.find_element_by_link_text(cafe)
                print("!!!!!", cafe_link.get_attribute('href'))
                list_cafe_link.append(cafe_link.get_attribute('href'))
        except:
            print ("Check your internet speed, Errors occur on {} page, Check your internet speed.".format(page)) 
            
        # Get a link to next page and click the link, then delay for 5 seconds for page loading.      
        try:                         
            next_page = driver.find_elements_by_xpath("//a[@class = 'lemon--a__373c0__IEZFH link__373c0__29943 next-link navigation-button__373c0__1D3Ug link-color--blue-dark__373c0__1mhJo link-size--default__373c0__1skgq']")[-1]
            next_page.click()        
            time.sleep(5)
        except:
            print (" Check your internet speed, Cannot Click to Next page, Stop scaping restaurant link")

        
    driver.quit()
     # Return a list of cafe links                   
    return list_cafe_link


# In[52]:


# def getRestaurant(location):
#     restaurantName = ""
#     finalDate = []
#     finalScore = []
#     finalReview = []
#     finalName = []
#     restaurantName = []
#     finalPriceRange = []
#     hasNextComment = True
#     basicUrl = "http://www.yelp.com/search?find_desc=Restaurants&find_loc="+location
#     newUrl = ""
#     k = 1    # limit for number of pages of restaurants to be scrapped
#              # The reason here for 3 pages is that yelp.com has a limit amount of scrappings per day to 7,500 
    
#     print("1.1")

#     driver = webdriver.Firefox(executable_path=executable_path)


#     driver.implicitly_wait(5)

#     driver.get(basicUrl+"&sortby=review_count")


    
    
#     # get customer reviews of each restaurant in the current page

#     # find all the restaurant links in the current page
#     # locate a "more" link by css selector
#     more_links = []
#     more_links=driver.\
#     find_elements_by_css_selector("a.lemon--a__373c0__IEZFH.link__373c0__29943.link-color--blue-dark__373c0__1mhJo.link-size--inherit__373c0__2JXk5")

#     for more_link in more_links:
#         # click the link of each restaurant
#         more_link.click()
        
#         # Now in the restaurant reviews page 1
#         # Save all customers' names into names[]
#         namePerPage = driver.\
#         find_elements_by_css_selector('a[data-analytics-label="about me"]')
        
#         for name in namePerPage:
#             finalName.append(name.text)
#             print(name.text)



#         # Find all the 


# In[57]:


# Function to Scrape reviews data
def getData(list_of_links):

    list_of_data = []

    
    print (list_of_links)    
    for link in list_of_links:
        print (link)
        executable_path = '/Users/charleschen/Downloads/geckodriver'
        driver = webdriver.Firefox(executable_path=executable_path)
        driver.implicitly_wait(10)
        driver.get(link)
        time.sleep(5)
        try:
        # Get Cafe name
            cafe_name = driver.find_elements_by_xpath("//meta[@itemprop = 'name']")[1].get_attribute('content')
            print(cafe_name)
            # Get rating from average and individual
            rating_list = driver.find_elements_by_xpath("//meta[@itemprop = 'ratingValue']")    
            # Get average_rating
            average_rating = rating_list[0].get_attribute('content')
            
            # Get JSON file from Xpath, and use json.loads to get latitude and longtitude as a dict
            get_lat_and_long = driver.find_elements_by_xpath("//div[@class= 'lightbox-map hidden']")
            latitude_longitude = json.loads(get_lat_and_long[0].get_attribute('data-map-state'))['center']
            latitude = latitude_longitude['latitude']
            longtitude = latitude_longitude['longitude']
        except:
            print ("Check your internet speed, Error occurs on the link {}, Skip this restaurant".format(link))
        
        # Loop each review page.  
        pages = 0
        while pages<34:
            try:
                
                # Get a review list.
                review_list = driver.find_elements_by_xpath("//p[@lang = 'en']")
                # Get an individual rating list
                individual_rating = rating_list[1:]
                # Get a review date list.
                review_date = driver.find_elements_by_xpath("//span[@class= 'rating-qualifier']")

            
                # Get each review, rating, and review date,  then put everything in a tuple
                # and append the tuple to the list.
                for index in range(len(review_list)):
                                
                    review = review_list[index].text.replace('\n', ' ')
                    rating = individual_rating[index].get_attribute('content')
                    date = review_date[index].text.split()[0]
                    
                    list_of_data.append((cafe_name, average_rating, review, rating, date, latitude, longtitude))                
            except:                
                print ("Check your internet speed, Error on this review page occurs, Skip the page")
                
            # Click to next review page and wait for 5 seconds for page to load. if there's no next page, error gets made,                 
            # and break the while loop.
            try:
                next_page = driver.find_element_by_xpath("//a[@class = 'u-decoration-none next pagination-links_anchor']")       
                pages += 1
                print("Page "+ pages)
                next_page.click()        
                time.sleep(5)
            except:
                break
        # Close Firefox browser
        driver.quit()
    
    # Create colum name for dataframe.
    columns = ['cafe_name', 'avg_rating', 'review', 'indv_rating', 'review_date', 'latitude', 'longtitude']
    # Create a data frame for all data.
    df = pd.DataFrame(list_of_data, columns = columns)
    
    df.to_csv("Review data.csv", index=False)


# In[58]:


if __name__ == "__main__":

   # Test Q1
    list_cafe_link=getRestaurant(1)
    
    getData(list_cafe_link)
    #print(data)
    # out of getRestaurant() !!!
    #print("out of getRestaurant() !!!")
    #Test Q2
    #plot_data(data)

    # Test Q3
#     data=getFullData("titanic")
#     print(len(data), data[0])
#     plot_data(data)


# In[ ]:




