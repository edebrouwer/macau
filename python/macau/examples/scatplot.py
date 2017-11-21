import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
colors= iter(cm.rainbow(np.linspace(0,1,rmse_mat.shape[0])))
fig,ax=plt.subplots()
absx=np.mean(cens_numbers,axis=1)
for idx in range(rmse_mat.shape[0]):
    ax.scatter(rmse_mat[idx],rmse_mat2[idx],label='Censoring Prop : '+str(absx[idx]),color=next(colors))
ax.plot([0,6],[0,6])
#ax.plot(absx,np.mean(rmse_mat,axis=1)+2*np.std(rmse_mat,axis=1),'k--',color="green")
#ax.plot(absx,np.mean(rmse_mat,axis=1)-2*np.std(rmse_mat,axis=1),'k--',color="green")
#ax.plot(absx,np.mean(rmse_mat2,axis=1),'k-',label='Normal Regression')
#ax.plot(absx,np.mean(rmse_mat2,axis=1)+2*np.std(rmse_mat2,axis=1),'k--')
#ax.plot(absx,np.mean(rmse_mat2,axis=1)-2*np.std(rmse_mat2,axis=1),'k--')
legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.title('Standard and Censored RMSE (5loops) for different censoring proportions. 500x1 matrix-0zeros-test(0.2)')
plt.xlabel('RMSE Censored Macau')
plt.ylabel('RMSE Standard Macau')
plt.show()
