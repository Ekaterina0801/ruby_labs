require 'open-uri'
require 'nokogiri'
puts Nokogiri::HTML.parse(URI.open('https://news.sky.com/story/wreck-of-sister-vessel-to-famous-17th-century-vasa-warship-found-in-sweden-12730450')).text
doc = Nokogiri::HTML(URI.open('https://news.sky.com/story/wreck-of-sister-vessel-to-famous-17th-century-vasa-warship-found-in-sweden-12730450'))
puts doc.to_html