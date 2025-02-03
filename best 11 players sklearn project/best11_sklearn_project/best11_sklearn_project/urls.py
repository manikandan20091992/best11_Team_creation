"""
URL configuration for best11_sklearn_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from best11_sklearn_app import views

urlpatterns = [
    path('select_best_11/', views.select_best_11_players, name='select_best_11'),

    path('ai_team_best11/', views.ai_team_best11, name='ai_team_best11'),

    path('chatgpt_canvas_best11/', views.chatgpt_canvas_best11, name='chatgpt_canvas_best11'),#using dataset excel

    path('cricketRealDataAiteam/', views.cricketRealDataAiteam, name='submit_selected_players'),#using realdata
    path('seriesMatchesListApi/', views.seriesMatchesListApi),
    path('players_squad_list/<str:match_id>', views.players_squad_list,name='match_detail'),




    



]
