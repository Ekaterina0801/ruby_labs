# Функция сортирует команды в турнирной таблице. За победу дается одно турнирное очко. Ничьих не бывает.
# Если у команд одинаковое количество турнирных очков, то они сортируются по разнице забитых и пропущенных очков за турнир.
# Если и этот показатель не помог определить порядок команд, нужно выбросить исключение.
# Формат данных:
# {
#   home: 'LAL',
#   away: 'CLE',
#   score: '120:115'
# },
#   {
#     home: 'MIA',
#     away: 'SAC',
#     score: '111:102'
#   },
#   {
#     home: 'SAC',
#     away: 'LAL',
#     score: '98:114'
#   },
#   {
#     home: 'CLE',
#     away: 'MIA',
#     score: '100:96'
#   }


class InvalidRes<RuntimeError; end

def rank_teams(matches)
  arr={}
  matches.each do
  |match|
    home = match[:home]
    away = match[:away]
    score = match[:score]
    res = score.split(':')
    res1 = res[0].to_i;
    res2 = res[1].to_i;
    if res1>res2
      if arr.has_key?(home)
        m = arr.dig(home)
        m[:total]+=1
        m[:goals]+=res1;
        m[:lose]+=res2;
        m[:difference]=m[:goals]-m[:lose]
      else
        arr.merge!(home=>{total:1,goals:res1,lose:res2, difference:res1-res2})
      end
      if arr.has_key?(away)
        m = arr.dig(away)
        m[:goals]+=res2
        m[:lose]+=res1
        m[:difference]=m[:goals]-m[:lose]
      else
        arr.merge!(away=>{total:0,goals:res2,lose:res1,difference:res2-res1})
      end
    else
      if arr.has_key?(away)
        m = arr.dig(away)
        m[:total]+=1;
        m[:goals]+=res2;
        m[:lose]+=res1;
        m[:difference]=m[:goals]-m[:lose]
      else
        arr.merge!(away=>{total:1,goals:res2,lose:res1, difference:res2-res1})
      end
      if arr.has_key?(home)
        m = arr.dig(home)
        m[:goals]+=res1;
        m[:lose]+=res2;
        m[:difference]=m[:goals]-m[:lose]
      else
        arr.merge!(home=>{total:0,goals:res1,lose:res2, difference:res1-res2})
      end
      end
  end
  arr = arr.sort_by{|_,value| value.values_at(:total, :difference)}
  arr = arr.reverse
  result = [];
  dif=0;
  m=0;
  arr.each do
    |key,value|
    result.push(key)
    if m==value[:total] and dif==value[:difference]
      throw InvalidRes
    end
    dif=value[:difference]
    m = value[:total]
  end
  result
end

#p rank_teams([{home:"MIA",away:"LAL",score:"100:105"},{home:"LAL",away:"MIA",score:"100:105"}])
p rank_teams([{home:"MIA",away:"LAL",score:"100:105"}])
p rank_teams([{ home: 'LAL', away: 'CLE', score: '120:115' }, { home: 'MIA', away: 'SAC', score: '111:102' },
             { home: 'SAC', away: 'LAL', score: '98:114' }, { home: 'CLE', away: 'MIA', score: '100:96' }])